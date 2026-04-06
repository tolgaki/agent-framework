// Copyright (c) Microsoft. All rights reserved.

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use reqwest_eventsource::{Event, EventSource, RequestBuilderExt};
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;
use tracing::debug;

use agent_framework_core::client::ChatClient;
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::http_limits::DEFAULT_MAX_RESPONSE_BYTES;
use agent_framework_core::redact::scrub_error_body;
use agent_framework_core::secret::SecretString;
use agent_framework_core::streaming::{AbortOnDrop, AbortingStream, ResponseStream};
use agent_framework_core::types::{
    ChatOptions, ChatResponse, ChatResponseUpdate, Content, FinishReason, Message, Role, ToolCallUpdate, ToolChoice,
    Usage,
};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_API_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Configuration for the Anthropic chat client.
#[derive(Clone, Debug)]
pub struct AnthropicConfig {
    /// Anthropic API key (redacted in `Debug`).
    pub api_key: SecretString,

    /// Model identifier (e.g., "claude-sonnet-4-20250514").
    pub model: String,

    /// Maximum tokens to generate. Defaults to 4096.
    pub max_tokens: u32,

    /// Base URL for the API. Defaults to `https://api.anthropic.com`.
    pub base_url: String,

    /// Overall request timeout. Defaults to 120 seconds.
    pub request_timeout: Duration,

    /// TCP connect timeout. Defaults to 10 seconds.
    pub connect_timeout: Duration,

    /// `anthropic-version` header value. Defaults to a known-good pin; update
    /// when Anthropic publishes a newer version you want to opt into.
    pub api_version: String,
}

impl AnthropicConfig {
    /// Create a config from environment variables.
    ///
    /// Reads `ANTHROPIC_API_KEY` (required) and optionally `ANTHROPIC_MODEL`.
    pub fn from_env() -> AgentResult<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| AgentError::InvalidRequest("ANTHROPIC_API_KEY environment variable is not set".to_string()))?;
        let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string());

        Ok(Self {
            api_key: SecretString::new(api_key),
            model,
            max_tokens: DEFAULT_MAX_TOKENS,
            base_url: DEFAULT_BASE_URL.to_string(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            api_version: DEFAULT_API_VERSION.to_string(),
        })
    }
}

/// A [`ChatClient`] implementation for the Anthropic Messages API.
pub struct AnthropicChatClient {
    config: AnthropicConfig,
    http: reqwest::Client,
}

impl AnthropicChatClient {
    /// Create a new Anthropic chat client.
    ///
    /// Returns an error if the underlying TLS backend cannot be initialised.
    pub fn new(config: AnthropicConfig) -> AgentResult<Self> {
        // `Policy::none()` is deliberate: reqwest's default redirect policy
        // strips `Authorization` / `Cookie` on cross-origin hops, but custom
        // headers like `x-api-key` are NOT stripped. Following any redirect
        // could leak the Anthropic API key to a third-party host.
        let http = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .connect_timeout(config.connect_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| AgentError::InvalidRequest(format!("failed to build HTTP client: {e}")))?;
        Ok(Self { config, http })
    }

    /// Build the request body for the Anthropic Messages API.
    fn build_request_body(&self, messages: &[Message], options: Option<&ChatOptions>) -> AgentResult<AnthropicRequest> {
        let model = options.and_then(|o| o.model.as_deref()).unwrap_or(&self.config.model);

        let max_tokens = options.and_then(|o| o.max_tokens).unwrap_or(self.config.max_tokens);

        // Extract system messages.
        let system: Option<String> = {
            let system_texts: Vec<&str> = messages
                .iter()
                .filter(|m| m.role == Role::System)
                .flat_map(|m| m.content.iter())
                .filter_map(|c| c.as_text())
                .collect();
            if system_texts.is_empty() {
                None
            } else {
                Some(system_texts.join("\n\n"))
            }
        };

        // Convert non-system messages to Anthropic format.
        //
        // Anthropic's Messages API requires strict user/assistant alternation.
        // Our framework emits one `Role::Tool` message per tool call, and both
        // Tool and User roles map to Anthropic's "user" role — left alone, that
        // produces consecutive user messages which the API rejects. Merge any
        // consecutive same-role entries by concatenating their content blocks.
        let api_messages: Vec<AnthropicMessage> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(message_to_anthropic)
            .fold(Vec::<AnthropicMessage>::new(), |mut acc, msg| {
                match acc.last_mut() {
                    Some(prev) if prev.role == msg.role => prev.content.extend(msg.content),
                    _ => acc.push(msg),
                }
                acc
            });

        // Convert tool definitions (if any) to Anthropic's wire format.
        let tools: Option<Vec<AnthropicTool>> = options.and_then(|o| {
            if o.tools.is_empty() {
                None
            } else {
                Some(
                    o.tools
                        .iter()
                        .map(|d| AnthropicTool {
                            name: d.name.clone(),
                            description: d.description.clone(),
                            input_schema: d.parameters_schema.clone(),
                        })
                        .collect(),
                )
            }
        });

        // Map tool_choice to Anthropic's shape. `None` means "forbid tool
        // calls" which Anthropic expresses by sending no tools at all.
        let (anthropic_tool_choice, tools) = match options.and_then(|o| o.tool_choice.as_ref()) {
            Some(ToolChoice::None) => (None, None),
            Some(ToolChoice::Auto) => (Some(serde_json::json!({"type": "auto"})), tools),
            Some(ToolChoice::Required) => (Some(serde_json::json!({"type": "any"})), tools),
            Some(ToolChoice::Specific { name }) => (Some(serde_json::json!({"type": "tool", "name": name})), tools),
            None => (None, tools),
        };

        Ok(AnthropicRequest {
            model: model.to_string(),
            max_tokens,
            system,
            messages: api_messages,
            tools,
            tool_choice: anthropic_tool_choice,
            temperature: options.and_then(|o| o.temperature),
            top_p: options.and_then(|o| o.top_p),
            top_k: options.and_then(|o| o.top_k),
            metadata: options
                .and_then(|o| o.user.clone())
                .map(|u| AnthropicMetadata { user_id: Some(u) }),
            stop_sequences: options.map(|o| o.stop_sequences.clone()).filter(|s| !s.is_empty()),
            stream: None,
        })
    }

    /// Build the HTTP headers used for every Anthropic API request.
    fn build_headers(&self) -> AgentResult<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let mut key_value = HeaderValue::from_str(self.config.api_key.expose())
            .map_err(|_| AgentError::InvalidRequest("ANTHROPIC_API_KEY contains invalid characters".to_string()))?;
        key_value.set_sensitive(true);
        headers.insert("x-api-key", key_value);
        let version = HeaderValue::from_str(&self.config.api_version)
            .map_err(|_| AgentError::InvalidRequest("anthropic_version contains invalid characters".to_string()))?;
        headers.insert("anthropic-version", version);
        Ok(headers)
    }
}

#[async_trait]
impl ChatClient for AnthropicChatClient {
    async fn get_response(&self, messages: &[Message], options: Option<&ChatOptions>) -> AgentResult<ChatResponse> {
        let body = self.build_request_body(messages, options)?;
        let url = format!("{}/v1/messages", self.config.base_url);

        debug!(model = %body.model, "Sending request to Anthropic");

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
            let error_text = response.text().await.unwrap_or_default();
            return Err(AgentError::provider(
                format!("Anthropic API error ({status}): {}", scrub_error_body(&error_text)),
                Some(status.as_u16()),
            ));
        }

        let bytes = read_bounded_body(response, DEFAULT_MAX_RESPONSE_BYTES).await?;
        let api_response: AnthropicResponse = serde_json::from_slice(&bytes)?;
        Ok(anthropic_response_to_chat_response(api_response))
    }

    fn get_response_stream(&self, messages: &[Message], options: Option<&ChatOptions>) -> AgentResult<ResponseStream> {
        let mut body = self.build_request_body(messages, options)?;
        body.stream = Some(true);
        let url = format!("{}/v1/messages", self.config.base_url);

        debug!(model = %body.model, "Opening Anthropic stream");

        let request = self.http.post(&url).headers(self.build_headers()?).json(&body);

        let event_source = request
            .eventsource()
            .map_err(|e| AgentError::HttpError(format!("failed to open event source: {e}")))?;

        Ok(spawn_anthropic_stream(event_source))
    }
}

// ---- Anthropic API types ----

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<AnthropicMetadata>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize)]
struct AnthropicMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    user_id: Option<String>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
enum AnthropicContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult { tool_use_id: String, content: String },
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    id: String,
    content: Vec<AnthropicContent>,
    stop_reason: Option<String>,
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// ---- Conversion functions ----

fn message_to_anthropic(msg: &Message) -> AnthropicMessage {
    let role = match msg.role {
        Role::User | Role::Tool => "user",
        Role::Assistant => "assistant",
        Role::System => "user", // Should not reach here (filtered above).
    };

    let content: Vec<AnthropicContent> = msg
        .content
        .iter()
        .map(|c| match c {
            Content::Text { text } => AnthropicContent::Text { text: text.clone() },
            Content::ToolCall { id, name, arguments } => AnthropicContent::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: arguments.clone(),
            },
            Content::ToolResult { tool_call_id, content } => AnthropicContent::ToolResult {
                tool_use_id: tool_call_id.clone(),
                content: content.clone(),
            },
            Content::Data { media_type, .. } => AnthropicContent::Text {
                text: format!("[binary data: {media_type}]"),
            },
            Content::Uri { uri, .. } => AnthropicContent::Text {
                text: format!("[uri: {uri}]"),
            },
        })
        .collect();

    AnthropicMessage {
        role: role.to_string(),
        content,
    }
}

/// Read an HTTP response body into memory with a hard byte cap.
///
/// Checks `Content-Length` first, then streams bytes and rejects if the
/// running total exceeds `max_bytes`. Prevents OOM from an attacker-controlled
/// `base_url` or a compromised upstream.
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

/// Per-block state tracked during a streaming response so we can attribute
/// `input_json_delta` events to the originating tool call by its block index.
#[derive(Debug)]
struct StreamBlockState {
    tool_id: Option<String>,
    tool_name: Option<String>,
}

/// Spawn a task that drains an Anthropic SSE stream and forwards
/// [`ChatResponseUpdate`] values into an mpsc channel. The returned
/// [`ResponseStream`] yields values as they arrive, and the spawned task
/// is aborted as soon as the stream is dropped.
fn spawn_anthropic_stream(mut event_source: EventSource) -> ResponseStream {
    let (tx, rx) = tokio::sync::mpsc::channel::<AgentResult<ChatResponseUpdate>>(16);
    let handle = tokio::spawn(async move {
        let mut blocks: HashMap<u32, StreamBlockState> = HashMap::new();

        while let Some(event) = event_source.next().await {
            match event {
                Ok(Event::Open) => continue,
                Ok(Event::Message(msg)) => {
                    match handle_anthropic_event(&msg.event, &msg.data, &mut blocks) {
                        Ok(Some(update)) => {
                            if tx.send(Ok(update)).await.is_err() {
                                // Receiver was dropped; stop reading.
                                break;
                            }
                        }
                        Ok(None) => {}
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

/// Convert a single Anthropic SSE event into an optional `ChatResponseUpdate`.
///
/// Returns `None` for events (ping, content_block_start text with empty text,
/// message_start, content_block_stop, message_stop) that don't produce an
/// outward update but that may update internal block state.
fn handle_anthropic_event(
    event_name: &str,
    data: &str,
    blocks: &mut HashMap<u32, StreamBlockState>,
) -> AgentResult<Option<ChatResponseUpdate>> {
    match event_name {
        "message_start" | "ping" | "content_block_stop" | "message_stop" => Ok(None),
        "content_block_start" => {
            let parsed: BlockStartEvent = serde_json::from_str(data)?;
            match parsed.content_block {
                AnthropicContent::ToolUse { id, name, .. } => {
                    blocks.insert(
                        parsed.index,
                        StreamBlockState {
                            tool_id: Some(id.clone()),
                            tool_name: Some(name.clone()),
                        },
                    );
                    // Emit an initial tool-call update carrying the id+name so
                    // downstream consumers can wire up accumulators before any
                    // argument deltas arrive.
                    Ok(Some(ChatResponseUpdate {
                        text: None,
                        tool_call: Some(ToolCallUpdate {
                            id,
                            name,
                            arguments_delta: String::new(),
                        }),
                        finish_reason: None,
                        usage: None,
                    }))
                }
                _ => {
                    blocks.insert(
                        parsed.index,
                        StreamBlockState {
                            tool_id: None,
                            tool_name: None,
                        },
                    );
                    Ok(None)
                }
            }
        }
        "content_block_delta" => {
            let parsed: BlockDeltaEvent = serde_json::from_str(data)?;
            match parsed.delta {
                AnthropicDelta::TextDelta { text } => Ok(Some(ChatResponseUpdate {
                    text: Some(text),
                    tool_call: None,
                    finish_reason: None,
                    usage: None,
                })),
                AnthropicDelta::InputJsonDelta { partial_json } => {
                    let state = blocks.get(&parsed.index);
                    let (id, name) = match state {
                        Some(s) if s.tool_id.is_some() => (
                            s.tool_id.clone().unwrap_or_default(),
                            s.tool_name.clone().unwrap_or_default(),
                        ),
                        _ => {
                            // Out-of-order delta with no known block; surface
                            // as a warning and drop this update rather than
                            // fabricating a tool-call id.
                            debug!(index = parsed.index, "input_json_delta for unknown block; dropping");
                            return Ok(None);
                        }
                    };
                    Ok(Some(ChatResponseUpdate {
                        text: None,
                        tool_call: Some(ToolCallUpdate {
                            id,
                            name,
                            arguments_delta: partial_json,
                        }),
                        finish_reason: None,
                        usage: None,
                    }))
                }
            }
        }
        "message_delta" => {
            let parsed: MessageDeltaEvent = serde_json::from_str(data)?;
            let finish_reason = parsed.delta.stop_reason.as_deref().map(map_stop_reason);
            let usage = parsed.usage.map(|u| Usage {
                // message_delta only includes output_tokens; preserve input as 0.
                input_tokens: 0,
                output_tokens: u.output_tokens,
            });
            if finish_reason.is_none() && usage.is_none() {
                Ok(None)
            } else {
                Ok(Some(ChatResponseUpdate {
                    text: None,
                    tool_call: None,
                    finish_reason,
                    usage,
                }))
            }
        }
        "error" => {
            let parsed: ErrorEvent = serde_json::from_str(data)?;
            Err(AgentError::provider(
                format!(
                    "Anthropic stream error ({}): {}",
                    parsed.error.error_type, parsed.error.message
                ),
                None,
            ))
        }
        _ => {
            // Unknown event types are forward-compatible: silently ignore.
            debug!(event = event_name, "Unknown Anthropic SSE event; ignoring");
            Ok(None)
        }
    }
}

#[derive(Deserialize)]
struct BlockStartEvent {
    index: u32,
    content_block: AnthropicContent,
}

#[derive(Deserialize)]
struct BlockDeltaEvent {
    index: u32,
    delta: AnthropicDelta,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum AnthropicDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

#[derive(Deserialize)]
struct MessageDeltaEvent {
    delta: MessageDelta,
    usage: Option<StreamUsage>,
}

#[derive(Deserialize)]
struct MessageDelta {
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct StreamUsage {
    output_tokens: u32,
}

#[derive(Deserialize)]
struct ErrorEvent {
    error: ErrorPayload,
}

#[derive(Deserialize)]
struct ErrorPayload {
    #[serde(rename = "type")]
    error_type: String,
    message: String,
}

fn map_stop_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" => FinishReason::MaxTokens,
        "tool_use" => FinishReason::ToolUse,
        _ => FinishReason::Stop,
    }
}

fn anthropic_response_to_chat_response(resp: AnthropicResponse) -> ChatResponse {
    let mut content_items = Vec::new();

    for block in &resp.content {
        match block {
            AnthropicContent::Text { text } => {
                content_items.push(Content::text(text));
            }
            AnthropicContent::ToolUse { id, name, input } => {
                content_items.push(Content::tool_call(id, name, input.clone()));
            }
            AnthropicContent::ToolResult { .. } => {
                // Tool results in response are unusual; skip.
            }
        }
    }

    let finish_reason = resp.stop_reason.as_deref().map(map_stop_reason);

    let usage = resp.usage.map(|u| Usage {
        input_tokens: u.input_tokens,
        output_tokens: u.output_tokens,
    });

    let message = Message {
        role: Role::Assistant,
        content: content_items,
        name: None,
        metadata: HashMap::new(),
    };

    ChatResponse {
        messages: vec![message],
        response_id: Some(resp.id),
        finish_reason,
        usage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_client() -> AnthropicChatClient {
        AnthropicChatClient::new(AnthropicConfig {
            api_key: SecretString::new("sk-test"),
            model: "claude-test".into(),
            max_tokens: 100,
            base_url: "https://example.invalid".into(),
            request_timeout: std::time::Duration::from_secs(10),
            connect_timeout: std::time::Duration::from_secs(1),
            api_version: DEFAULT_API_VERSION.to_string(),
        })
        .expect("client builds in tests")
    }

    #[test]
    fn consecutive_tool_result_messages_are_merged_into_one_user_message() {
        // Regression: the agent loop emits one Role::Tool message per tool call.
        // Both Tool and User roles map to Anthropic's "user" role, and left
        // alone this produces consecutive user messages which Anthropic rejects.
        let client = make_client();
        let messages = vec![
            Message::user("run some tools"),
            Message::tool_result("call_1", "result 1"),
            Message::tool_result("call_2", "result 2"),
        ];
        let body = client.build_request_body(&messages, None).unwrap();

        // We expect exactly one merged user message containing all three blocks:
        // the original user text plus both tool_results.
        assert_eq!(body.messages.len(), 1, "consecutive user/tool messages must merge");
        assert_eq!(body.messages[0].role, "user");
        assert_eq!(body.messages[0].content.len(), 3);
    }

    #[test]
    fn system_messages_are_hoisted_into_system_field() {
        let client = make_client();
        let messages = vec![
            Message::system("you are helpful"),
            Message::system("and concise"),
            Message::user("hi"),
        ];
        let body = client.build_request_body(&messages, None).unwrap();
        assert_eq!(body.system.as_deref(), Some("you are helpful\n\nand concise"));
        assert_eq!(body.messages.len(), 1);
    }

    #[test]
    fn alternating_user_assistant_pattern_preserved() {
        let client = make_client();
        let messages = vec![Message::user("hi"), Message::assistant("hello"), Message::user("bye")];
        let body = client.build_request_body(&messages, None).unwrap();
        assert_eq!(body.messages.len(), 3);
        assert_eq!(body.messages[0].role, "user");
        assert_eq!(body.messages[1].role, "assistant");
        assert_eq!(body.messages[2].role, "user");
    }

    #[test]
    fn chat_options_tools_are_serialized() {
        use agent_framework_core::tools::ToolDefinition;

        let client = make_client();
        let options = ChatOptions {
            tools: vec![ToolDefinition::no_params("search", "Search the web")],
            ..Default::default()
        };
        let body = client
            .build_request_body(&[Message::user("hi")], Some(&options))
            .unwrap();
        let tools = body.tools.expect("tools should be serialized");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "search");
    }

    #[test]
    fn stop_reason_mapping() {
        assert_eq!(map_stop_reason("end_turn"), FinishReason::Stop);
        assert_eq!(map_stop_reason("stop_sequence"), FinishReason::Stop);
        assert_eq!(map_stop_reason("max_tokens"), FinishReason::MaxTokens);
        assert_eq!(map_stop_reason("tool_use"), FinishReason::ToolUse);
        // Unknown reasons fall through to Stop.
        assert_eq!(map_stop_reason("new_reason_from_future"), FinishReason::Stop);
    }

    #[test]
    fn stream_text_delta_event() {
        let mut blocks = HashMap::new();
        let update = handle_anthropic_event(
            "content_block_delta",
            r#"{"index":0,"delta":{"type":"text_delta","text":"hello"}}"#,
            &mut blocks,
        )
        .unwrap()
        .expect("text_delta should produce an update");
        assert_eq!(update.text.as_deref(), Some("hello"));
    }

    #[test]
    fn stream_tool_use_block_start_registers_and_emits_initial_update() {
        let mut blocks = HashMap::new();
        let update = handle_anthropic_event(
            "content_block_start",
            r#"{"index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup","input":{}}}"#,
            &mut blocks,
        )
        .unwrap()
        .expect("tool_use block_start should emit an initial update");
        assert!(blocks.contains_key(&1));
        let tc = update.tool_call.as_ref().expect("should carry a tool call");
        assert_eq!(tc.id, "toolu_1");
        assert_eq!(tc.name, "lookup");
        assert_eq!(tc.arguments_delta, "");
    }

    #[test]
    fn stream_input_json_delta_uses_block_index_state() {
        let mut blocks = HashMap::new();
        // First register the block.
        handle_anthropic_event(
            "content_block_start",
            r#"{"index":0,"content_block":{"type":"tool_use","id":"toolu_a","name":"calc","input":{}}}"#,
            &mut blocks,
        )
        .unwrap();
        // Then deliver a json delta referencing index=0.
        let update = handle_anthropic_event(
            "content_block_delta",
            r#"{"index":0,"delta":{"type":"input_json_delta","partial_json":"{\"x\":1}"}}"#,
            &mut blocks,
        )
        .unwrap()
        .expect("input_json_delta should produce an update");
        let tc = update.tool_call.as_ref().unwrap();
        assert_eq!(tc.id, "toolu_a");
        assert_eq!(tc.name, "calc");
        assert_eq!(tc.arguments_delta, "{\"x\":1}");
    }
}
