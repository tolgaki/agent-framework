// Copyright (c) Microsoft. All rights reserved.

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use futures::stream::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;
use tracing::debug;

use agent_framework_core::client::ChatClient;
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::http_limits::DEFAULT_MAX_RESPONSE_BYTES;
use agent_framework_core::redact::{scrub_error_body, MAX_ERROR_BODY_LEN};
use agent_framework_core::secret::SecretString;
use agent_framework_core::streaming::{AbortOnDrop, AbortingStream, ResponseStream};
use agent_framework_core::types::{
    ChatOptions, ChatResponse, ChatResponseUpdate, Content, FinishReason, Message, ResponseFormat, Role,
    ToolCallUpdate, ToolChoice, Usage,
};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Configuration for the OpenAI chat client.
#[derive(Clone, Debug)]
pub struct OpenAIConfig {
    /// OpenAI API key (redacted in `Debug`).
    pub api_key: SecretString,

    /// Model identifier (e.g., "gpt-4o").
    pub model: String,

    /// Maximum tokens to generate.
    pub max_tokens: u32,

    /// Base URL for the API.
    pub base_url: String,

    /// Overall request timeout. Defaults to 120 seconds.
    pub request_timeout: Duration,

    /// TCP connect timeout. Defaults to 10 seconds.
    pub connect_timeout: Duration,
}

impl OpenAIConfig {
    /// Create a config from environment variables.
    ///
    /// Reads `OPENAI_API_KEY` (required) and optionally `OPENAI_MODEL`.
    pub fn from_env() -> AgentResult<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| AgentError::InvalidRequest("OPENAI_API_KEY environment variable is not set".to_string()))?;
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());

        Ok(Self {
            api_key: SecretString::new(api_key),
            model,
            max_tokens: DEFAULT_MAX_TOKENS,
            base_url: DEFAULT_BASE_URL.to_string(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        })
    }
}

/// A [`ChatClient`] implementation for the OpenAI Chat Completions API.
pub struct OpenAIChatClient {
    config: OpenAIConfig,
    http: reqwest::Client,
}

impl OpenAIChatClient {
    /// Create a new OpenAI chat client.
    ///
    /// Returns an error if the underlying TLS backend cannot be initialised.
    pub fn new(config: OpenAIConfig) -> AgentResult<Self> {
        // `Policy::none()` is deliberate: reqwest's default redirect policy
        // strips `Authorization` / `Cookie` on cross-origin hops, but a
        // misconfigured `base_url` could still lead the `Authorization`
        // header to an attacker-controlled host via a same-origin bounce.
        // API calls should never follow redirects.
        let http = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .connect_timeout(config.connect_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| AgentError::InvalidRequest(format!("failed to build HTTP client: {e}")))?;
        Ok(Self { config, http })
    }

    /// Build HTTP headers (Authorization + JSON content type) used for every request.
    fn build_headers(&self) -> AgentResult<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let mut auth = HeaderValue::from_str(&format!("Bearer {}", self.config.api_key.expose()))
            .map_err(|_| AgentError::InvalidRequest("OPENAI_API_KEY contains invalid characters".to_string()))?;
        auth.set_sensitive(true);
        headers.insert(AUTHORIZATION, auth);
        Ok(headers)
    }

    fn build_request_body(&self, messages: &[Message], options: Option<&ChatOptions>) -> OpenAIRequest {
        let model = options
            .and_then(|o| o.model.as_deref())
            .unwrap_or(&self.config.model)
            .to_string();

        let max_tokens = options.and_then(|o| o.max_tokens).unwrap_or(self.config.max_tokens);

        let api_messages: Vec<OpenAIMessage> = messages.iter().flat_map(message_to_openai).collect();

        // Convert tool definitions (if any) to OpenAI's wire format.
        let tools: Option<Vec<OpenAITool>> = options.and_then(|o| {
            if o.tools.is_empty() {
                None
            } else {
                Some(
                    o.tools
                        .iter()
                        .map(|d| OpenAITool {
                            r#type: "function".to_string(),
                            function: OpenAIFunction {
                                name: d.name.clone(),
                                description: d.description.clone(),
                                parameters: d.parameters_schema.clone(),
                            },
                        })
                        .collect(),
                )
            }
        });

        OpenAIRequest {
            model,
            max_tokens: Some(max_tokens),
            messages: api_messages,
            tools,
            temperature: options.and_then(|o| o.temperature),
            top_p: options.and_then(|o| o.top_p),
            seed: options.and_then(|o| o.seed),
            frequency_penalty: options.and_then(|o| o.frequency_penalty),
            presence_penalty: options.and_then(|o| o.presence_penalty),
            logit_bias: options.map(|o| o.logit_bias.clone()).filter(|b| !b.is_empty()),
            user: options.and_then(|o| o.user.clone()),
            response_format: options
                .and_then(|o| o.response_format.as_ref())
                .map(openai_response_format),
            tool_choice: options.and_then(|o| o.tool_choice.as_ref()).map(openai_tool_choice),
            stop: options.map(|o| o.stop_sequences.clone()).filter(|s| !s.is_empty()),
            stream: None,
            stream_options: None,
        }
    }
}

/// Convert a framework [`ResponseFormat`] to the OpenAI wire shape.
fn openai_response_format(fmt: &ResponseFormat) -> serde_json::Value {
    match fmt {
        ResponseFormat::Text => serde_json::json!({"type": "text"}),
        ResponseFormat::JsonObject => serde_json::json!({"type": "json_object"}),
        ResponseFormat::JsonSchema { name, schema, strict } => {
            let mut obj = serde_json::json!({
                "type": "json_schema",
                "json_schema": { "name": name, "schema": schema }
            });
            if let Some(s) = strict {
                obj["json_schema"]["strict"] = serde_json::json!(*s);
            }
            obj
        }
    }
}

/// Convert a framework [`ToolChoice`] to the OpenAI wire shape.
fn openai_tool_choice(choice: &ToolChoice) -> serde_json::Value {
    match choice {
        ToolChoice::Auto => serde_json::json!("auto"),
        ToolChoice::None => serde_json::json!("none"),
        ToolChoice::Required => serde_json::json!("required"),
        ToolChoice::Specific { name } => serde_json::json!({
            "type": "function",
            "function": { "name": name }
        }),
    }
}

#[async_trait]
impl ChatClient for OpenAIChatClient {
    async fn get_response(&self, messages: &[Message], options: Option<&ChatOptions>) -> AgentResult<ChatResponse> {
        let body = self.build_request_body(messages, options);
        let url = format!("{}/chat/completions", self.config.base_url);

        debug!(model = %body.model, "Sending request to OpenAI");

        let response = self
            .http
            .post(&url)
            .headers(self.build_headers()?)
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentError::HttpError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_bytes = read_bounded_body(response, MAX_ERROR_BODY_LEN).await.unwrap_or_default();
            let error_text = String::from_utf8_lossy(&error_bytes);
            return Err(AgentError::provider(
                format!("OpenAI API error ({status}): {}", scrub_error_body(&error_text)),
                Some(status.as_u16()),
            ));
        }

        let bytes = read_bounded_body(response, DEFAULT_MAX_RESPONSE_BYTES).await?;
        let api_response: OpenAIResponse = serde_json::from_slice(&bytes)?;
        Ok(openai_response_to_chat_response(api_response))
    }

    async fn get_response_stream(&self, messages: &[Message], options: Option<&ChatOptions>) -> AgentResult<ResponseStream> {
        let mut body = self.build_request_body(messages, options);
        body.stream = Some(true);
        // Ask OpenAI to include usage in the final chunk.
        body.stream_options = Some(StreamOptions { include_usage: true });
        let url = format!("{}/chat/completions", self.config.base_url);

        debug!(model = %body.model, "Opening OpenAI stream");

        let request = self.http.post(&url).headers(self.build_headers()?).json(&body);

        let event_source = request
            .eventsource()
            .map_err(|e| AgentError::HttpError(format!("failed to open event source: {e}")))?;

        Ok(spawn_openai_stream(event_source))
    }
}

// ---- OpenAI API types ----

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Serialize)]
struct OpenAIMessage {
    role: String,
    /// Content can be a JSON string, an array of content parts (for multimodal),
    /// or null. Using `serde_json::Value` covers all three cases.
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
struct OpenAIToolCall {
    id: String,
    r#type: String,
    function: OpenAIFunctionCall,
}

#[derive(Serialize, Deserialize, Clone)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct OpenAITool {
    r#type: String,
    function: OpenAIFunction,
}

#[derive(Serialize)]
struct OpenAIFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    id: String,
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIResponseMessage {
    #[allow(dead_code)]
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Deserialize, Clone)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

// ---- Conversion functions ----

fn message_to_openai(msg: &Message) -> Vec<OpenAIMessage> {
    let role = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };

    // Tool-role messages: OpenAI requires one message per tool_call_id.
    // Emit a separate OpenAIMessage for each ToolResult.
    if msg.role == Role::Tool {
        let results: Vec<_> = msg
            .content
            .iter()
            .filter_map(|c| {
                if let Content::ToolResult { tool_call_id, content } = c {
                    Some(OpenAIMessage {
                        role: "tool".to_string(),
                        content: Some(serde_json::Value::String(content.clone())),
                        tool_calls: None,
                        tool_call_id: Some(tool_call_id.clone()),
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
        return vec![OpenAIMessage {
            role: "tool".to_string(),
            content: None,
            tool_calls: None,
            tool_call_id: None,
        }];
    }

    // Collect tool calls from assistant messages.
    let tool_calls: Vec<_> = msg
        .content
        .iter()
        .filter_map(|c| {
            if let Content::ToolCall { id, name, arguments } = c {
                Some(OpenAIToolCall {
                    id: id.clone(),
                    r#type: "function".to_string(),
                    function: OpenAIFunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(arguments).unwrap_or_default(),
                    },
                })
            } else {
                None
            }
        })
        .collect();

    // Build the content field. Use an array of content parts if the message
    // contains multimodal items (images), otherwise a plain string.
    let has_multimodal = msg.content.iter().any(|c| matches!(c, Content::Data { .. } | Content::Uri { .. }));
    let content = if has_multimodal {
        let parts: Vec<serde_json::Value> = msg
            .content
            .iter()
            .filter_map(|c| match c {
                Content::Text { text } => Some(serde_json::json!({"type": "text", "text": text})),
                Content::Data { data, media_type } => {
                    let b64 = BASE64_STANDARD.encode(data);
                    let data_url = format!("data:{media_type};base64,{b64}");
                    Some(serde_json::json!({"type": "image_url", "image_url": {"url": data_url}}))
                }
                Content::Uri { uri, .. } => {
                    Some(serde_json::json!({"type": "image_url", "image_url": {"url": uri}}))
                }
                _ => None,
            })
            .collect();
        if parts.is_empty() {
            None
        } else {
            Some(serde_json::Value::Array(parts))
        }
    } else {
        let text = msg.text();
        if text.is_empty() { None } else { Some(serde_json::Value::String(text)) }
    };

    vec![OpenAIMessage {
        role: role.to_string(),
        content,
        tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
        tool_call_id: None,
    }]
}

/// Read an HTTP response body into memory with a hard byte cap.
async fn read_bounded_body(response: reqwest::Response, max_bytes: usize) -> AgentResult<Vec<u8>> {
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

// ---- Streaming ----

/// Spawn a task that drains an OpenAI SSE stream and forwards
/// [`ChatResponseUpdate`] values into an mpsc channel. The spawned task
/// is aborted as soon as the returned [`ResponseStream`] is dropped.
fn spawn_openai_stream(mut event_source: EventSource) -> ResponseStream {
    let (tx, rx) = tokio::sync::mpsc::channel::<AgentResult<ChatResponseUpdate>>(16);
    let handle = tokio::spawn(async move {
        // Tool calls stream as deltas keyed by `index`; we track id+name so we
        // can attach them to each arguments-delta update.
        let mut tool_call_index: HashMap<u32, (String, String)> = HashMap::new();

        while let Some(event) = event_source.next().await {
            match event {
                Ok(Event::Open) => continue,
                Ok(Event::Message(msg)) => {
                    // OpenAI signals end-of-stream with a literal `[DONE]` payload.
                    if msg.data == "[DONE]" {
                        break;
                    }
                    match handle_openai_chunk(&msg.data, &mut tool_call_index) {
                        Ok(updates) => {
                            for update in updates {
                                if tx.send(Ok(update)).await.is_err() {
                                    event_source.close();
                                    return;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            break;
                        }
                    }
                }
                Err(reqwest_eventsource::Error::StreamEnded) => break,
                Err(e) => {
                    let _ = tx
                        .send(Err(AgentError::HttpError(format!("SSE stream error: {e}"))))
                        .await;
                    break;
                }
            }
        }
        event_source.close();
    });

    let guard = AbortOnDrop::new(handle.abort_handle());
    ResponseStream::new(AbortingStream::new(ReceiverStream::new(rx), guard))
}

/// Convert a single OpenAI chunk into zero or more `ChatResponseUpdate`s.
fn handle_openai_chunk(
    data: &str,
    tool_call_index: &mut HashMap<u32, (String, String)>,
) -> AgentResult<Vec<ChatResponseUpdate>> {
    let chunk: OpenAIStreamChunk = serde_json::from_str(data)?;
    let mut updates = Vec::new();

    // Usage arrives on the final chunk (when stream_options.include_usage=true).
    // That chunk has an empty `choices` array, so handle it before the choice loop.
    if let Some(usage) = chunk.usage {
        updates.push(ChatResponseUpdate {
            text: None,
            tool_call: None,
            finish_reason: None,
            usage: Some(Usage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
            }),
        });
    }

    for choice in chunk.choices.into_iter() {
        if let Some(content) = choice.delta.content {
            if !content.is_empty() {
                updates.push(ChatResponseUpdate {
                    text: Some(content),
                    tool_call: None,
                    finish_reason: None,
                    usage: None,
                });
            }
        }

        if let Some(tool_calls) = choice.delta.tool_calls {
            for tc in tool_calls {
                // First chunk for a given index carries id + function.name.
                // Subsequent chunks carry additional argument fragments.
                if let (Some(id), Some(function)) = (tc.id.clone(), tc.function.as_ref()) {
                    let name = function.name.clone().unwrap_or_default();
                    tool_call_index.insert(tc.index, (id, name));
                }
                let (id, name) = tool_call_index
                    .get(&tc.index)
                    .cloned()
                    .unwrap_or_else(|| (String::new(), String::new()));
                let arguments_delta = tc.function.and_then(|f| f.arguments).unwrap_or_default();
                updates.push(ChatResponseUpdate {
                    text: None,
                    tool_call: Some(ToolCallUpdate {
                        id,
                        name,
                        arguments_delta,
                    }),
                    finish_reason: None,
                    usage: None,
                });
            }
        }

        if let Some(reason) = choice.finish_reason {
            let mapped = match reason.as_str() {
                "stop" => FinishReason::Stop,
                "length" => FinishReason::MaxTokens,
                "tool_calls" => FinishReason::ToolUse,
                "content_filter" => FinishReason::ContentFilter,
                _ => FinishReason::Stop,
            };
            updates.push(ChatResponseUpdate {
                text: None,
                tool_call: None,
                finish_reason: Some(mapped),
                usage: None,
            });
        }
    }

    Ok(updates)
}

#[derive(Deserialize)]
struct OpenAIStreamChunk {
    #[serde(default)]
    choices: Vec<OpenAIStreamChoice>,
    #[serde(default)]
    usage: Option<OpenAIUsage>,
}

#[derive(Deserialize)]
struct OpenAIStreamChoice {
    delta: OpenAIDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct OpenAIDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAIStreamToolCall>>,
}

#[derive(Deserialize)]
struct OpenAIStreamToolCall {
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAIStreamFunctionCall>,
}

#[derive(Deserialize)]
struct OpenAIStreamFunctionCall {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

fn openai_response_to_chat_response(resp: OpenAIResponse) -> ChatResponse {
    let choice = resp.choices.into_iter().next();

    let (message, finish_reason_str) = match choice {
        Some(c) => (c.message, c.finish_reason),
        None => {
            return ChatResponse {
                messages: vec![],
                response_id: Some(resp.id),
                finish_reason: None,
                usage: None,
            };
        }
    };

    let mut content_items = Vec::new();

    if let Some(text) = message.content {
        content_items.push(Content::text(text));
    }

    if let Some(tool_calls) = message.tool_calls {
        for tc in tool_calls {
            // Preserve the raw argument string if the model returns invalid
            // JSON, rather than silently becoming Null. The agent tool loop
            // will surface an error to the model for malformed arguments.
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or_else(|_| serde_json::Value::String(tc.function.arguments.clone()));
            content_items.push(Content::tool_call(&tc.id, &tc.function.name, args));
        }
    }

    let finish_reason = finish_reason_str.as_deref().map(|r| match r {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::MaxTokens,
        "tool_calls" => FinishReason::ToolUse,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    });

    let usage = resp.usage.map(|u| Usage {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
    });

    let msg = Message {
        role: Role::Assistant,
        content: content_items,
        name: None,
        metadata: HashMap::new(),
    };

    ChatResponse {
        messages: vec![msg],
        response_id: Some(resp.id),
        finish_reason,
        usage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::types::Content;

    #[test]
    fn tool_role_message_populates_content_field() {
        let msg = Message::tool_result("call_123", "the sky is blue");
        let msgs = message_to_openai(&msg);
        assert_eq!(msgs.len(), 1);
        let oa = &msgs[0];
        assert_eq!(oa.role, "tool");
        assert_eq!(oa.tool_call_id.as_deref(), Some("call_123"));
        assert_eq!(oa.content.as_ref().and_then(|v| v.as_str()), Some("the sky is blue"));
        assert!(oa.tool_calls.is_none());
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
        let msgs = message_to_openai(&msg);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(msgs[1].tool_call_id.as_deref(), Some("call_2"));
    }

    #[test]
    fn assistant_message_with_tool_calls_serializes_correctly() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                Content::text("let me check"),
                Content::tool_call("call_1", "get_weather", serde_json::json!({"city": "Seattle"})),
            ],
            name: None,
            metadata: HashMap::new(),
        };
        let msgs = message_to_openai(&msg);
        assert_eq!(msgs.len(), 1);
        let oa = &msgs[0];
        assert_eq!(oa.role, "assistant");
        assert_eq!(oa.content.as_ref().and_then(|v| v.as_str()), Some("let me check"));
        let calls = oa.tool_calls.as_ref().expect("tool_calls should be present");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("Seattle"));
    }

    #[test]
    fn user_message_text_goes_to_content() {
        let msg = Message::user("hello");
        let msgs = message_to_openai(&msg);
        assert_eq!(msgs.len(), 1);
        let oa = &msgs[0];
        assert_eq!(oa.role, "user");
        assert_eq!(oa.content.as_ref().and_then(|v| v.as_str()), Some("hello"));
        assert!(oa.tool_calls.is_none());
        assert!(oa.tool_call_id.is_none());
    }

    #[test]
    fn tools_from_chat_options_serialize() {
        use agent_framework_core::tools::ToolDefinition;

        let config = OpenAIConfig {
            api_key: SecretString::new("sk-test"),
            model: "gpt-4o".into(),
            max_tokens: 100,
            base_url: "https://example.invalid".into(),
            request_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(1),
        };
        let client = OpenAIChatClient::new(config).expect("client builds in tests");

        let options = ChatOptions {
            tools: vec![ToolDefinition::no_params("ping", "Ping").unwrap()],
            ..Default::default()
        };

        let body = client.build_request_body(&[Message::user("hi")], Some(&options));
        let tools = body.tools.expect("tools should be serialized");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "ping");
    }

    #[test]
    fn stream_chunk_accumulates_tool_calls() {
        let mut index = HashMap::new();
        // First chunk has id + name + partial args.
        let first = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_x","function":{"name":"lookup","arguments":"{\"q"}}]}, "finish_reason":null}]}"#;
        let updates = handle_openai_chunk(first, &mut index).unwrap();
        assert_eq!(updates.len(), 1);
        let tc = updates[0].tool_call.as_ref().unwrap();
        assert_eq!(tc.id, "call_x");
        assert_eq!(tc.name, "lookup");
        assert_eq!(tc.arguments_delta, "{\"q");

        // Continuation chunk only has the index + more arg chars.
        let second = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\":1}"}}]}, "finish_reason":null}]}"#;
        let updates = handle_openai_chunk(second, &mut index).unwrap();
        let tc = updates[0].tool_call.as_ref().unwrap();
        assert_eq!(tc.id, "call_x");
        assert_eq!(tc.name, "lookup");
        assert_eq!(tc.arguments_delta, "\":1}");

        // Final chunk: tool_calls finish reason.
        let done = r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#;
        let updates = handle_openai_chunk(done, &mut index).unwrap();
        assert!(updates.iter().any(|u| u.finish_reason == Some(FinishReason::ToolUse)));
    }
}
