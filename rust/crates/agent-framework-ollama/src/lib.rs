// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework Ollama
//!
//! Ollama local LLM provider for the Microsoft Agent Framework.
//!
//! This crate provides [`OllamaChatClient`], a [`ChatClient`](agent_framework_core::ChatClient)
//! implementation that communicates with a locally running [Ollama](https://ollama.com) instance
//! via its `/api/chat` endpoint.

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;
use tracing::debug;

use agent_framework_core::client::ChatClient;
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::http_limits::DEFAULT_MAX_RESPONSE_BYTES;
use agent_framework_core::redact::{scrub_error_body, MAX_ERROR_BODY_LEN};
use agent_framework_core::streaming::{AbortOnDrop, AbortingStream, ResponseStream};
use agent_framework_core::types::{
    ChatOptions, ChatResponse, ChatResponseUpdate, Content, FinishReason, Message, Role,
    ToolCallUpdate, Usage,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const DEFAULT_BASE_URL: &str = "http://localhost:11434";
const DEFAULT_MODEL: &str = "llama3.2";
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Configuration for the Ollama chat client.
#[derive(Clone, Debug)]
pub struct OllamaConfig {
    /// Base URL of the Ollama server (e.g., `http://localhost:11434`).
    pub base_url: String,

    /// Model identifier (e.g., `"llama3.2"`).
    pub model: String,

    /// Overall request timeout. Defaults to 300 seconds (local models are
    /// slower than cloud APIs).
    pub request_timeout: Duration,

    /// TCP connect timeout. Defaults to 10 seconds.
    pub connect_timeout: Duration,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }
}

impl OllamaConfig {
    /// Create a config from environment variables.
    ///
    /// Reads `OLLAMA_BASE_URL` (optional, defaults to `http://localhost:11434`)
    /// and `OLLAMA_MODEL` (optional, defaults to `llama3.2`).
    pub fn from_env() -> Self {
        let base_url =
            std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

        Self {
            base_url,
            model,
            ..Self::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// A [`ChatClient`] implementation for a locally running Ollama instance.
pub struct OllamaChatClient {
    config: OllamaConfig,
    http: reqwest::Client,
}

impl OllamaChatClient {
    /// Create a new Ollama chat client.
    ///
    /// Returns an error if the underlying HTTP client cannot be built.
    pub fn new(config: OllamaConfig) -> AgentResult<Self> {
        let http = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .connect_timeout(config.connect_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| {
                AgentError::InvalidRequest(format!("failed to build HTTP client: {e}"))
            })?;
        Ok(Self { config, http })
    }

    /// Build the Ollama request body.
    fn build_request_body(
        &self,
        messages: &[Message],
        options: Option<&ChatOptions>,
        stream: bool,
    ) -> OllamaRequest {
        let model = options
            .and_then(|o| o.model.as_deref())
            .unwrap_or(&self.config.model)
            .to_string();

        let api_messages: Vec<OllamaMessage> =
            messages.iter().flat_map(message_to_ollama).collect();

        let tools: Option<Vec<OllamaTool>> = options.and_then(|o| {
            if o.tools.is_empty() {
                None
            } else {
                Some(
                    o.tools
                        .iter()
                        .map(|d| OllamaTool {
                            r#type: "function".to_string(),
                            function: OllamaFunction {
                                name: d.name.clone(),
                                description: d.description.clone(),
                                parameters: d.parameters_schema.clone(),
                            },
                        })
                        .collect(),
                )
            }
        });

        let mut request_options = OllamaOptions::default();
        if let Some(opts) = options {
            request_options.temperature = opts.temperature;
            request_options.top_p = opts.top_p;
            request_options.top_k = opts.top_k;
            request_options.seed = opts.seed.map(|s| s as i32);
            request_options.num_predict = opts.max_tokens.map(|t| t as i32);
            if !opts.stop_sequences.is_empty() {
                request_options.stop = Some(opts.stop_sequences.clone());
            }
            request_options.frequency_penalty = opts.frequency_penalty;
            request_options.presence_penalty = opts.presence_penalty;
        }

        let has_options = request_options.temperature.is_some()
            || request_options.top_p.is_some()
            || request_options.top_k.is_some()
            || request_options.seed.is_some()
            || request_options.num_predict.is_some()
            || request_options.stop.is_some()
            || request_options.frequency_penalty.is_some()
            || request_options.presence_penalty.is_some();

        OllamaRequest {
            model,
            messages: api_messages,
            stream,
            tools,
            options: if has_options {
                Some(request_options)
            } else {
                None
            },
        }
    }
}

#[async_trait]
impl ChatClient for OllamaChatClient {
    async fn get_response(
        &self,
        messages: &[Message],
        options: Option<&ChatOptions>,
    ) -> AgentResult<ChatResponse> {
        let body = self.build_request_body(messages, options, false);
        let url = format!("{}/api/chat", self.config.base_url);

        debug!(model = %body.model, "Sending request to Ollama");

        let response = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentError::HttpError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_bytes = read_bounded_body(response, MAX_ERROR_BODY_LEN)
                .await
                .unwrap_or_default();
            let error_text = String::from_utf8_lossy(&error_bytes);
            return Err(AgentError::provider(
                format!(
                    "Ollama API error ({status}): {}",
                    scrub_error_body(&error_text)
                ),
                Some(status.as_u16()),
            ));
        }

        let bytes = read_bounded_body(response, DEFAULT_MAX_RESPONSE_BYTES).await?;
        let api_response: OllamaResponse = serde_json::from_slice(&bytes)?;
        Ok(ollama_response_to_chat_response(api_response))
    }

    async fn get_response_stream(
        &self,
        messages: &[Message],
        options: Option<&ChatOptions>,
    ) -> AgentResult<ResponseStream> {
        let body = self.build_request_body(messages, options, true);
        let url = format!("{}/api/chat", self.config.base_url);

        debug!(model = %body.model, "Opening Ollama stream");

        let response = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentError::HttpError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_bytes = read_bounded_body(response, MAX_ERROR_BODY_LEN)
                .await
                .unwrap_or_default();
            let error_text = String::from_utf8_lossy(&error_bytes);
            return Err(AgentError::provider(
                format!(
                    "Ollama API error ({status}): {}",
                    scrub_error_body(&error_text)
                ),
                Some(status.as_u16()),
            ));
        }

        Ok(spawn_ollama_stream(response))
    }
}

// ---------------------------------------------------------------------------
// Ollama API types (internal)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OllamaTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct OllamaMessage {
    role: String,
    #[serde(default)]
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct OllamaToolCall {
    function: OllamaFunctionCall,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct OllamaFunctionCall {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Serialize)]
struct OllamaTool {
    r#type: String,
    function: OllamaFunction,
}

#[derive(Serialize)]
struct OllamaFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Serialize, Default)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
}

/// Full (non-streaming) response from Ollama's `/api/chat`.
#[derive(Deserialize)]
struct OllamaResponse {
    message: OllamaMessage,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    done_reason: Option<String>,
}

/// A single NDJSON chunk from Ollama's streaming `/api/chat`.
#[derive(Deserialize)]
struct OllamaStreamChunk {
    message: OllamaMessage,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    done_reason: Option<String>,
}

// ---------------------------------------------------------------------------
// Conversion: framework types -> Ollama types
// ---------------------------------------------------------------------------

/// Convert a framework [`Message`] to one or more Ollama messages.
///
/// Tool result messages are mapped back to role `"tool"` with the result text
/// as content, matching Ollama's expected format.
fn message_to_ollama(msg: &Message) -> Vec<OllamaMessage> {
    let role = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };

    // Tool-role messages: Ollama expects one message per tool result.
    if msg.role == Role::Tool {
        let results: Vec<_> = msg
            .content
            .iter()
            .filter_map(|c| {
                if let Content::ToolResult { content, .. } = c {
                    Some(OllamaMessage {
                        role: "tool".to_string(),
                        content: content.clone(),
                        tool_calls: None,
                    })
                } else {
                    None
                }
            })
            .collect();
        if !results.is_empty() {
            return results;
        }
        // Fallback: tool message with no ToolResult content.
        return vec![OllamaMessage {
            role: "tool".to_string(),
            content: String::new(),
            tool_calls: None,
        }];
    }

    // Collect tool calls from assistant messages.
    let tool_calls: Vec<OllamaToolCall> = msg
        .content
        .iter()
        .filter_map(|c| {
            if let Content::ToolCall {
                name, arguments, ..
            } = c
            {
                Some(OllamaToolCall {
                    function: OllamaFunctionCall {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                })
            } else {
                None
            }
        })
        .collect();

    let text = msg.text();

    vec![OllamaMessage {
        role: role.to_string(),
        content: text,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
    }]
}

// ---------------------------------------------------------------------------
// Conversion: Ollama types -> framework types
// ---------------------------------------------------------------------------

fn ollama_response_to_chat_response(resp: OllamaResponse) -> ChatResponse {
    let mut content_items = Vec::new();

    if !resp.message.content.is_empty() {
        content_items.push(Content::text(&resp.message.content));
    }

    if let Some(tool_calls) = resp.message.tool_calls {
        for tc in tool_calls {
            content_items.push(Content::tool_call(
                // Ollama does not provide tool call IDs; generate a synthetic one.
                generate_tool_call_id(&tc.function.name),
                &tc.function.name,
                tc.function.arguments,
            ));
        }
    }

    let finish_reason = if resp.done {
        match resp.done_reason.as_deref() {
            Some("stop") | None => {
                // If we emitted tool calls, the finish reason is ToolUse.
                if content_items
                    .iter()
                    .any(|c| matches!(c, Content::ToolCall { .. }))
                {
                    Some(FinishReason::ToolUse)
                } else {
                    Some(FinishReason::Stop)
                }
            }
            Some("length") => Some(FinishReason::MaxTokens),
            _ => Some(FinishReason::Stop),
        }
    } else {
        None
    };

    let usage = match (resp.prompt_eval_count, resp.eval_count) {
        (Some(input), Some(output)) => Some(Usage {
            input_tokens: input,
            output_tokens: output,
        }),
        _ => None,
    };

    let msg = Message {
        role: Role::Assistant,
        content: content_items,
        name: None,
        metadata: HashMap::new(),
    };

    ChatResponse {
        messages: vec![msg],
        response_id: None,
        finish_reason,
        usage,
    }
}

/// Generate a synthetic tool call ID since Ollama does not provide one.
fn generate_tool_call_id(name: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("ollama_{name}_{id}")
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// Spawn a task that reads Ollama's NDJSON stream and forwards
/// [`ChatResponseUpdate`] values into an mpsc channel. The spawned task
/// is aborted as soon as the returned [`ResponseStream`] is dropped.
fn spawn_ollama_stream(response: reqwest::Response) -> ResponseStream {
    let (tx, rx) = tokio::sync::mpsc::channel::<AgentResult<ChatResponseUpdate>>(16);

    let handle = tokio::spawn(async move {
        let mut byte_stream = response.bytes_stream();
        let mut buf = Vec::new();
        let mut accumulated_tool_calls: Vec<OllamaToolCall> = Vec::new();

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx
                        .send(Err(AgentError::HttpError(format!(
                            "stream read error: {e}"
                        ))))
                        .await;
                    return;
                }
            };

            buf.extend_from_slice(&chunk);

            // Process complete NDJSON lines from the buffer.
            while let Some(newline_pos) = buf.iter().position(|&b| b == b'\n') {

                let line: Vec<u8> = buf.drain(..=newline_pos).collect();
                let line_str = String::from_utf8_lossy(&line);
                let trimmed = line_str.trim();
                if trimmed.is_empty() {
                    continue;
                }

                let stream_chunk: OllamaStreamChunk = match serde_json::from_str(trimmed) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx
                            .send(Err(AgentError::InvalidResponse(format!(
                                "failed to parse Ollama stream chunk: {e}"
                            ))))
                            .await;
                        return;
                    }
                };

                let is_done = stream_chunk.done;
                let updates = handle_ollama_chunk(&stream_chunk, &mut accumulated_tool_calls);
                for update in updates {
                    if tx.send(Ok(update)).await.is_err() {
                        return;
                    }
                }

                if is_done {
                    return;
                }
            }
        }

        // Process any remaining data in the buffer (final line without trailing newline).
        if !buf.is_empty() {
            let line_str = String::from_utf8_lossy(&buf);
            let trimmed = line_str.trim();
            if !trimmed.is_empty() {
                if let Ok(stream_chunk) = serde_json::from_str::<OllamaStreamChunk>(trimmed) {
                    let updates =
                        handle_ollama_chunk(&stream_chunk, &mut accumulated_tool_calls);
                    for update in updates {
                        if tx.send(Ok(update)).await.is_err() {
                            return;
                        }
                    }
                }
            }
        }
    });

    let guard = AbortOnDrop::new(handle.abort_handle());
    ResponseStream::new(AbortingStream::new(ReceiverStream::new(rx), guard))
}

/// Convert a single Ollama stream chunk into zero or more `ChatResponseUpdate`s.
fn handle_ollama_chunk(
    chunk: &OllamaStreamChunk,
    accumulated_tool_calls: &mut Vec<OllamaToolCall>,
) -> Vec<ChatResponseUpdate> {
    let mut updates = Vec::new();

    // Text content delta.
    if !chunk.message.content.is_empty() {
        updates.push(ChatResponseUpdate {
            text: Some(chunk.message.content.clone()),
            tool_call: None,
            finish_reason: None,
            usage: None,
        });
    }

    // Tool calls -- Ollama delivers them complete in a single chunk rather than
    // as incremental deltas (unlike OpenAI). We emit each as a ToolCallUpdate
    // with the full arguments.
    if let Some(tool_calls) = &chunk.message.tool_calls {
        for tc in tool_calls {
            accumulated_tool_calls.push(tc.clone());
            let id = generate_tool_call_id(&tc.function.name);
            let args_str = serde_json::to_string(&tc.function.arguments).unwrap_or_default();
            updates.push(ChatResponseUpdate {
                text: None,
                tool_call: Some(ToolCallUpdate {
                    id,
                    name: tc.function.name.clone(),
                    arguments_delta: args_str,
                }),
                finish_reason: None,
                usage: None,
            });
        }
    }

    // Final chunk.
    if chunk.done {
        let has_tool_calls = !accumulated_tool_calls.is_empty();
        let finish_reason = match chunk.done_reason.as_deref() {
            Some("stop") | None => {
                if has_tool_calls {
                    FinishReason::ToolUse
                } else {
                    FinishReason::Stop
                }
            }
            Some("length") => FinishReason::MaxTokens,
            _ => FinishReason::Stop,
        };

        let usage = match (chunk.prompt_eval_count, chunk.eval_count) {
            (Some(input), Some(output)) => Some(Usage {
                input_tokens: input,
                output_tokens: output,
            }),
            _ => None,
        };

        updates.push(ChatResponseUpdate {
            text: None,
            tool_call: None,
            finish_reason: Some(finish_reason),
            usage,
        });
    }

    updates
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read an HTTP response body into memory with a hard byte cap.
async fn read_bounded_body(
    response: reqwest::Response,
    max_bytes: usize,
) -> AgentResult<Vec<u8>> {
    if let Some(len) = response.content_length() {
        if len as u128 > max_bytes as u128 {
            return Err(AgentError::HttpError(format!(
                "response Content-Length {len} exceeds limit {max_bytes}"
            )));
        }
    }
    let mut buf = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| AgentError::HttpError(e.to_string()))?;
        if buf.len().saturating_add(chunk.len()) > max_bytes {
            return Err(AgentError::HttpError(format!(
                "response body exceeded {max_bytes} bytes"
            )));
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::tools::ToolDefinition;
    use agent_framework_core::types::Content;

    // -- Config tests --

    #[test]
    fn default_config_has_expected_values() {
        let config = OllamaConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, "llama3.2");
        assert_eq!(config.request_timeout, Duration::from_secs(300));
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
    }

    #[test]
    fn from_env_uses_defaults_when_vars_unset() {
        // Clear any existing env vars for isolation.
        std::env::remove_var("OLLAMA_BASE_URL");
        std::env::remove_var("OLLAMA_MODEL");
        let config = OllamaConfig::from_env();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, "llama3.2");
    }

    // -- Message conversion tests --

    #[test]
    fn user_message_converts_correctly() {
        let msg = Message::user("hello");
        let ollama_msgs = message_to_ollama(&msg);
        assert_eq!(ollama_msgs.len(), 1);
        assert_eq!(ollama_msgs[0].role, "user");
        assert_eq!(ollama_msgs[0].content, "hello");
        assert!(ollama_msgs[0].tool_calls.is_none());
    }

    #[test]
    fn system_message_converts_correctly() {
        let msg = Message::system("you are helpful");
        let ollama_msgs = message_to_ollama(&msg);
        assert_eq!(ollama_msgs.len(), 1);
        assert_eq!(ollama_msgs[0].role, "system");
        assert_eq!(ollama_msgs[0].content, "you are helpful");
    }

    #[test]
    fn assistant_message_with_text_converts_correctly() {
        let msg = Message::assistant("sure, let me help");
        let ollama_msgs = message_to_ollama(&msg);
        assert_eq!(ollama_msgs.len(), 1);
        assert_eq!(ollama_msgs[0].role, "assistant");
        assert_eq!(ollama_msgs[0].content, "sure, let me help");
        assert!(ollama_msgs[0].tool_calls.is_none());
    }

    #[test]
    fn assistant_message_with_tool_calls_converts_correctly() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                Content::text("let me check"),
                Content::tool_call(
                    "call_1",
                    "get_weather",
                    serde_json::json!({"city": "Seattle"}),
                ),
            ],
            name: None,
            metadata: HashMap::new(),
        };
        let ollama_msgs = message_to_ollama(&msg);
        assert_eq!(ollama_msgs.len(), 1);
        let m = &ollama_msgs[0];
        assert_eq!(m.role, "assistant");
        assert_eq!(m.content, "let me check");
        let calls = m.tool_calls.as_ref().expect("tool_calls should be present");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(
            calls[0].function.arguments,
            serde_json::json!({"city": "Seattle"})
        );
    }

    #[test]
    fn tool_result_message_converts_correctly() {
        let msg = Message::tool_result("call_123", "the sky is blue");
        let ollama_msgs = message_to_ollama(&msg);
        assert_eq!(ollama_msgs.len(), 1);
        assert_eq!(ollama_msgs[0].role, "tool");
        assert_eq!(ollama_msgs[0].content, "the sky is blue");
    }

    #[test]
    fn multiple_tool_results_produce_multiple_messages() {
        let msg = Message {
            role: Role::Tool,
            content: vec![
                Content::tool_result("call_1", "result_1"),
                Content::tool_result("call_2", "result_2"),
            ],
            name: None,
            metadata: HashMap::new(),
        };
        let ollama_msgs = message_to_ollama(&msg);
        assert_eq!(ollama_msgs.len(), 2);
        assert_eq!(ollama_msgs[0].content, "result_1");
        assert_eq!(ollama_msgs[1].content, "result_2");
    }

    // -- Response conversion tests --

    #[test]
    fn ollama_text_response_converts() {
        let resp = OllamaResponse {
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: "Hello!".to_string(),
                tool_calls: None,
            },
            done: true,
            eval_count: Some(10),
            prompt_eval_count: Some(5),
            done_reason: Some("stop".to_string()),
        };
        let chat = ollama_response_to_chat_response(resp);
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].text(), "Hello!");
        assert_eq!(chat.finish_reason, Some(FinishReason::Stop));
        let usage = chat.usage.unwrap();
        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, 10);
    }

    #[test]
    fn ollama_tool_call_response_converts() {
        let resp = OllamaResponse {
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: String::new(),
                tool_calls: Some(vec![OllamaToolCall {
                    function: OllamaFunctionCall {
                        name: "get_weather".to_string(),
                        arguments: serde_json::json!({"city": "Paris"}),
                    },
                }]),
            },
            done: true,
            eval_count: Some(20),
            prompt_eval_count: Some(15),
            done_reason: Some("stop".to_string()),
        };
        let chat = ollama_response_to_chat_response(resp);
        assert_eq!(chat.messages.len(), 1);
        let calls = chat.messages[0].tool_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, "get_weather");
        assert_eq!(chat.finish_reason, Some(FinishReason::ToolUse));
    }

    #[test]
    fn ollama_length_finish_reason() {
        let resp = OllamaResponse {
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: "truncated".to_string(),
                tool_calls: None,
            },
            done: true,
            eval_count: None,
            prompt_eval_count: None,
            done_reason: Some("length".to_string()),
        };
        let chat = ollama_response_to_chat_response(resp);
        assert_eq!(chat.finish_reason, Some(FinishReason::MaxTokens));
    }

    // -- Request body tests --

    #[test]
    fn build_request_body_uses_config_model() {
        let config = OllamaConfig {
            model: "codellama".to_string(),
            ..OllamaConfig::default()
        };
        let client = OllamaChatClient::new(config).unwrap();
        let body = client.build_request_body(&[Message::user("hi")], None, false);
        assert_eq!(body.model, "codellama");
        assert!(!body.stream);
    }

    #[test]
    fn build_request_body_respects_option_overrides() {
        let config = OllamaConfig::default();
        let client = OllamaChatClient::new(config).unwrap();
        let opts = ChatOptions {
            model: Some("phi3".to_string()),
            temperature: Some(0.5),
            max_tokens: Some(100),
            ..Default::default()
        };
        let body = client.build_request_body(&[Message::user("hi")], Some(&opts), true);
        assert_eq!(body.model, "phi3");
        assert!(body.stream);
        let options = body.options.unwrap();
        assert_eq!(options.temperature, Some(0.5));
        assert_eq!(options.num_predict, Some(100));
    }

    #[test]
    fn build_request_body_includes_tools() {
        let config = OllamaConfig::default();
        let client = OllamaChatClient::new(config).unwrap();
        let opts = ChatOptions {
            tools: vec![ToolDefinition::no_params("ping", "Ping the server").unwrap()],
            ..Default::default()
        };
        let body = client.build_request_body(&[Message::user("hi")], Some(&opts), false);
        let tools = body.tools.expect("tools should be present");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "ping");
        assert_eq!(tools[0].r#type, "function");
    }

    #[test]
    fn build_request_body_omits_empty_options() {
        let config = OllamaConfig::default();
        let client = OllamaChatClient::new(config).unwrap();
        let opts = ChatOptions::default();
        let body = client.build_request_body(&[Message::user("hi")], Some(&opts), false);
        assert!(body.options.is_none());
        assert!(body.tools.is_none());
    }

    // -- Streaming chunk tests --

    #[test]
    fn stream_chunk_with_text() {
        let chunk = OllamaStreamChunk {
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: "Hello".to_string(),
                tool_calls: None,
            },
            done: false,
            eval_count: None,
            prompt_eval_count: None,
            done_reason: None,
        };
        let mut acc = Vec::new();
        let updates = handle_ollama_chunk(&chunk, &mut acc);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].text.as_deref(), Some("Hello"));
        assert!(updates[0].finish_reason.is_none());
    }

    #[test]
    fn stream_chunk_done_sends_finish_reason() {
        let chunk = OllamaStreamChunk {
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: String::new(),
                tool_calls: None,
            },
            done: true,
            eval_count: Some(42),
            prompt_eval_count: Some(10),
            done_reason: Some("stop".to_string()),
        };
        let mut acc = Vec::new();
        let updates = handle_ollama_chunk(&chunk, &mut acc);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].finish_reason, Some(FinishReason::Stop));
        let usage = updates[0].usage.as_ref().unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 42);
    }

    #[test]
    fn stream_chunk_with_tool_calls_sets_tool_use_finish() {
        let tool_chunk = OllamaStreamChunk {
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: String::new(),
                tool_calls: Some(vec![OllamaToolCall {
                    function: OllamaFunctionCall {
                        name: "search".to_string(),
                        arguments: serde_json::json!({"q": "rust"}),
                    },
                }]),
            },
            done: false,
            eval_count: None,
            prompt_eval_count: None,
            done_reason: None,
        };
        let mut acc = Vec::new();
        let updates = handle_ollama_chunk(&tool_chunk, &mut acc);
        assert_eq!(updates.len(), 1);
        let tc = updates[0].tool_call.as_ref().unwrap();
        assert_eq!(tc.name, "search");

        // Final done chunk should report ToolUse because we accumulated tool calls.
        let done_chunk = OllamaStreamChunk {
            message: OllamaMessage {
                role: "assistant".to_string(),
                content: String::new(),
                tool_calls: None,
            },
            done: true,
            eval_count: Some(30),
            prompt_eval_count: Some(20),
            done_reason: Some("stop".to_string()),
        };
        let updates = handle_ollama_chunk(&done_chunk, &mut acc);
        assert!(updates
            .iter()
            .any(|u| u.finish_reason == Some(FinishReason::ToolUse)));
    }

    // -- Serialization round-trip tests --

    #[test]
    fn ollama_request_serializes_correctly() {
        let req = OllamaRequest {
            model: "llama3.2".to_string(),
            messages: vec![OllamaMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
                tool_calls: None,
            }],
            stream: false,
            tools: None,
            options: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "llama3.2");
        assert_eq!(json["stream"], false);
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"], "hello");
        // tool_calls should be absent (not null) when None.
        assert!(json["messages"][0].get("tool_calls").is_none());
        // options should be absent when None.
        assert!(json.get("options").is_none());
    }

    #[test]
    fn ollama_response_deserializes() {
        let json = r#"{
            "model": "llama3.2",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Hi there!"},
            "done": true,
            "eval_count": 15,
            "prompt_eval_count": 8,
            "done_reason": "stop"
        }"#;
        let resp: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.message.content, "Hi there!");
        assert!(resp.done);
        assert_eq!(resp.eval_count, Some(15));
        assert_eq!(resp.prompt_eval_count, Some(8));
    }

    #[test]
    fn ollama_stream_chunk_deserializes() {
        let json = r#"{"model":"llama3.2","message":{"role":"assistant","content":"Hi"},"done":false}"#;
        let chunk: OllamaStreamChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.message.content, "Hi");
        assert!(!chunk.done);
    }

    #[test]
    fn generate_tool_call_id_is_unique() {
        let id1 = generate_tool_call_id("test");
        let id2 = generate_tool_call_id("test");
        assert_ne!(id1, id2);
        assert!(id1.starts_with("ollama_test_"));
    }

    #[test]
    fn client_constructs_successfully() {
        let config = OllamaConfig::default();
        let client = OllamaChatClient::new(config);
        assert!(client.is_ok());
    }
}
