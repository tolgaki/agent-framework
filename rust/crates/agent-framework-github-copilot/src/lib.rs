// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework GitHub Copilot
//!
//! GitHub Copilot Chat API provider for the Microsoft Agent Framework.
//!
//! Mirrors the .NET `Microsoft.Agents.AI.GitHub.Copilot` project. Exposes
//! [`GitHubCopilotChatClient`], a [`ChatClient`](agent_framework_core::ChatClient)
//! implementation that talks to the GitHub Copilot Chat API at
//! <https://api.githubcopilot.com/chat/completions>. The wire format is
//! OpenAI-compatible (the `chat.completions` schema), but the auth and
//! headers differ.
//!
//! # Auth
//!
//! The client uses a GitHub token via `Authorization: Bearer {token}`.
//! In addition to the token, GitHub requires two headers that identify
//! the Copilot integration:
//!
//! ```text
//! Editor-Version: vscode/1.85.0
//! Copilot-Integration-Id: vscode-chat
//! ```
//!
//! # Example
//!
//! ```no_run
//! use agent_framework_github_copilot::{GitHubCopilotChatClient, GitHubCopilotConfig};
//! use agent_framework_core::client::ChatClient;
//! use agent_framework_core::types::Message;
//!
//! # async fn run() -> agent_framework_core::error::AgentResult<()> {
//! let config = GitHubCopilotConfig::from_env()?;
//! let client = GitHubCopilotChatClient::new(config)?;
//! let response = client.get_response(&[Message::user("Hi")], None).await?;
//! println!("{}", response.messages[0].text());
//! # Ok(()) }
//! ```

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE, USER_AGENT};
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
    ChatOptions, ChatResponse, ChatResponseUpdate, Content, FinishReason, Message, Role,
    ToolCallUpdate, Usage,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const DEFAULT_BASE_URL: &str = "https://api.githubcopilot.com";
const DEFAULT_MODEL: &str = "gpt-4o";
const DEFAULT_EDITOR_VERSION: &str = "vscode/1.85.0";
const DEFAULT_INTEGRATION_ID: &str = "vscode-chat";
const DEFAULT_USER_AGENT: &str = "agent-framework-github-copilot/0.1";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Configuration for the GitHub Copilot chat client.
#[derive(Clone, Debug)]
pub struct GitHubCopilotConfig {
    /// GitHub token (redacted in `Debug`).
    pub github_token: SecretString,

    /// Model identifier (e.g. `"gpt-4o"`, `"claude-3.5-sonnet"`).
    pub model: String,

    /// Value sent as the `Editor-Version` header. Defaults to `vscode/1.85.0`.
    pub editor_version: String,

    /// Value sent as the `Copilot-Integration-Id` header. Defaults to `vscode-chat`.
    pub integration_id: String,

    /// Base URL for the Copilot API. Defaults to `https://api.githubcopilot.com`.
    pub base_url: String,

    /// Maximum tokens to generate when the caller does not override.
    pub max_tokens: u32,

    /// Overall request timeout.
    pub request_timeout: Duration,

    /// TCP connect timeout.
    pub connect_timeout: Duration,
}

impl GitHubCopilotConfig {
    /// Build a config with the given token, defaulting everything else.
    pub fn new(github_token: impl Into<SecretString>) -> Self {
        Self {
            github_token: github_token.into(),
            model: DEFAULT_MODEL.to_string(),
            editor_version: DEFAULT_EDITOR_VERSION.to_string(),
            integration_id: DEFAULT_INTEGRATION_ID.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }

    /// Create a config from environment variables.
    ///
    /// Reads `GITHUB_TOKEN` (required) and `GITHUB_COPILOT_MODEL` (optional,
    /// defaults to `gpt-4o`).
    pub fn from_env() -> AgentResult<Self> {
        let token = std::env::var("GITHUB_TOKEN")
            .map_err(|_| AgentError::InvalidRequest("GITHUB_TOKEN environment variable is not set".to_string()))?;
        let model = std::env::var("GITHUB_COPILOT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

        Ok(Self {
            github_token: SecretString::new(token),
            model,
            editor_version: DEFAULT_EDITOR_VERSION.to_string(),
            integration_id: DEFAULT_INTEGRATION_ID.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        })
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// A [`ChatClient`] implementation for the GitHub Copilot Chat API.
pub struct GitHubCopilotChatClient {
    config: GitHubCopilotConfig,
    http: reqwest::Client,
}

impl GitHubCopilotChatClient {
    /// Create a new GitHub Copilot chat client.
    ///
    /// Returns an error if the underlying TLS backend cannot be initialised.
    pub fn new(config: GitHubCopilotConfig) -> AgentResult<Self> {
        // `Policy::none()` mirrors the other provider clients: we never want
        // the `Authorization` header to travel across a redirect, even one
        // that appears same-origin.
        let http = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .connect_timeout(config.connect_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| AgentError::InvalidRequest(format!("failed to build HTTP client: {e}")))?;
        Ok(Self { config, http })
    }

    /// Build HTTP headers used for every request. Credential headers are
    /// marked sensitive so they do not leak into logs.
    fn build_headers(&self) -> AgentResult<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        headers.insert(USER_AGENT, HeaderValue::from_static(DEFAULT_USER_AGENT));

        let mut auth = HeaderValue::from_str(&format!("Bearer {}", self.config.github_token.expose()))
            .map_err(|_| AgentError::InvalidRequest("GITHUB_TOKEN contains invalid characters".to_string()))?;
        auth.set_sensitive(true);
        headers.insert(AUTHORIZATION, auth);

        // `Editor-Version` and `Copilot-Integration-Id` are required by the
        // Copilot API. They are not credentials but they identify the
        // integration; we still mark them opaque rather than sensitive.
        let editor_version = HeaderValue::from_str(&self.config.editor_version).map_err(|_| {
            AgentError::InvalidRequest("editor_version contains invalid header characters".to_string())
        })?;
        headers.insert(HeaderName::from_static("editor-version"), editor_version);

        let integration_id = HeaderValue::from_str(&self.config.integration_id).map_err(|_| {
            AgentError::InvalidRequest("integration_id contains invalid header characters".to_string())
        })?;
        headers.insert(HeaderName::from_static("copilot-integration-id"), integration_id);

        Ok(headers)
    }

    fn build_request_body(&self, messages: &[Message], options: Option<&ChatOptions>) -> CopilotRequest {
        let model = options
            .and_then(|o| o.model.as_deref())
            .unwrap_or(&self.config.model)
            .to_string();

        let max_tokens = options.and_then(|o| o.max_tokens).unwrap_or(self.config.max_tokens);

        let api_messages: Vec<CopilotMessage> = messages.iter().flat_map(message_to_copilot).collect();

        let tools: Option<Vec<CopilotTool>> = options.and_then(|o| {
            if o.tools.is_empty() {
                None
            } else {
                Some(
                    o.tools
                        .iter()
                        .map(|d| CopilotTool {
                            r#type: "function".to_string(),
                            function: CopilotFunction {
                                name: d.name.clone(),
                                description: d.description.clone(),
                                parameters: d.parameters_schema.clone(),
                            },
                        })
                        .collect(),
                )
            }
        });

        CopilotRequest {
            model,
            max_tokens: Some(max_tokens),
            messages: api_messages,
            tools,
            temperature: options.and_then(|o| o.temperature),
            top_p: options.and_then(|o| o.top_p),
            stop: options.map(|o| o.stop_sequences.clone()).filter(|s| !s.is_empty()),
            stream: None,
        }
    }
}

#[async_trait]
impl ChatClient for GitHubCopilotChatClient {
    async fn get_response(
        &self,
        messages: &[Message],
        options: Option<&ChatOptions>,
    ) -> AgentResult<ChatResponse> {
        let body = self.build_request_body(messages, options);
        let url = format!("{}/chat/completions", self.config.base_url);

        debug!(model = %body.model, "Sending request to GitHub Copilot");

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
            let error_bytes = read_bounded_body(response, MAX_ERROR_BODY_LEN)
                .await
                .unwrap_or_default();
            let error_text = String::from_utf8_lossy(&error_bytes);
            return Err(AgentError::provider(
                format!("GitHub Copilot API error ({status}): {}", scrub_error_body(&error_text)),
                Some(status.as_u16()),
            ));
        }

        let bytes = read_bounded_body(response, DEFAULT_MAX_RESPONSE_BYTES).await?;
        let api_response: CopilotResponse = serde_json::from_slice(&bytes)?;
        Ok(copilot_response_to_chat_response(api_response))
    }

    async fn get_response_stream(
        &self,
        messages: &[Message],
        options: Option<&ChatOptions>,
    ) -> AgentResult<ResponseStream> {
        let mut body = self.build_request_body(messages, options);
        body.stream = Some(true);
        let url = format!("{}/chat/completions", self.config.base_url);

        debug!(model = %body.model, "Opening GitHub Copilot stream");

        let request = self.http.post(&url).headers(self.build_headers()?).json(&body);

        let event_source = request
            .eventsource()
            .map_err(|e| AgentError::HttpError(format!("failed to open event source: {e}")))?;

        Ok(spawn_copilot_stream(event_source))
    }
}

// ---------------------------------------------------------------------------
// Wire format (OpenAI-compatible subset used by GitHub Copilot)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct CopilotRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    messages: Vec<CopilotMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<CopilotTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Serialize)]
struct CopilotMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<CopilotToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
struct CopilotToolCall {
    id: String,
    r#type: String,
    function: CopilotFunctionCall,
}

#[derive(Serialize, Deserialize, Clone)]
struct CopilotFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct CopilotTool {
    r#type: String,
    function: CopilotFunction,
}

#[derive(Serialize)]
struct CopilotFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize)]
struct CopilotResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    choices: Vec<CopilotChoice>,
    #[serde(default)]
    usage: Option<CopilotUsage>,
}

#[derive(Deserialize)]
struct CopilotChoice {
    message: CopilotResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct CopilotResponseMessage {
    #[allow(dead_code)]
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<CopilotToolCall>>,
}

#[derive(Deserialize, Clone)]
struct CopilotUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

fn message_to_copilot(msg: &Message) -> Vec<CopilotMessage> {
    let role = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };

    // Tool-role messages: one CopilotMessage per tool_call_id.
    if msg.role == Role::Tool {
        let results: Vec<_> = msg
            .content
            .iter()
            .filter_map(|c| {
                if let Content::ToolResult { tool_call_id, content } = c {
                    Some(CopilotMessage {
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
        return vec![CopilotMessage {
            role: "tool".to_string(),
            content: None,
            tool_calls: None,
            tool_call_id: None,
        }];
    }

    let tool_calls: Vec<_> = msg
        .content
        .iter()
        .filter_map(|c| {
            if let Content::ToolCall { id, name, arguments } = c {
                Some(CopilotToolCall {
                    id: id.clone(),
                    r#type: "function".to_string(),
                    function: CopilotFunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(arguments).unwrap_or_default(),
                    },
                })
            } else {
                None
            }
        })
        .collect();

    let text = msg.text();
    let content = if text.is_empty() {
        None
    } else {
        Some(serde_json::Value::String(text))
    };

    vec![CopilotMessage {
        role: role.to_string(),
        content,
        tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
        tool_call_id: None,
    }]
}

fn copilot_response_to_chat_response(resp: CopilotResponse) -> ChatResponse {
    let choice = resp.choices.into_iter().next();

    let (message, finish_reason_str) = match choice {
        Some(c) => (c.message, c.finish_reason),
        None => {
            return ChatResponse {
                messages: vec![],
                response_id: resp.id,
                finish_reason: None,
                usage: None,
            };
        }
    };

    let mut content_items = Vec::new();
    if let Some(text) = message.content {
        if !text.is_empty() {
            content_items.push(Content::text(text));
        }
    }
    if let Some(tool_calls) = message.tool_calls {
        for tc in tool_calls {
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or_else(|_| serde_json::Value::String(tc.function.arguments.clone()));
            content_items.push(Content::tool_call(&tc.id, &tc.function.name, args));
        }
    }

    let finish_reason = finish_reason_str.as_deref().map(map_finish_reason);
    let usage = resp.usage.map(|u| Usage {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
    });

    let assistant = Message {
        role: Role::Assistant,
        content: content_items,
        name: None,
        metadata: HashMap::new(),
    };

    ChatResponse {
        messages: vec![assistant],
        response_id: resp.id,
        finish_reason,
        usage,
    }
}

fn map_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::MaxTokens,
        "tool_calls" => FinishReason::ToolUse,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

/// Spawn a task that drains an SSE stream and forwards [`ChatResponseUpdate`]
/// values into an mpsc channel. The spawned task is aborted as soon as the
/// returned [`ResponseStream`] is dropped.
fn spawn_copilot_stream(mut event_source: EventSource) -> ResponseStream {
    let (tx, rx) = tokio::sync::mpsc::channel::<AgentResult<ChatResponseUpdate>>(16);
    let handle = tokio::spawn(async move {
        // Tool calls stream as deltas keyed by `index`; we track id+name so we
        // can attach them to each arguments-delta update.
        let mut tool_call_index: HashMap<u32, (String, String)> = HashMap::new();

        while let Some(event) = event_source.next().await {
            match event {
                Ok(Event::Open) => continue,
                Ok(Event::Message(msg)) => {
                    if msg.data == "[DONE]" {
                        break;
                    }
                    match handle_copilot_chunk(&msg.data, &mut tool_call_index) {
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

/// Convert a single Copilot SSE chunk into zero or more `ChatResponseUpdate`s.
fn handle_copilot_chunk(
    data: &str,
    tool_call_index: &mut HashMap<u32, (String, String)>,
) -> AgentResult<Vec<ChatResponseUpdate>> {
    let chunk: CopilotStreamChunk = serde_json::from_str(data)?;
    let mut updates = Vec::new();

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
            updates.push(ChatResponseUpdate {
                text: None,
                tool_call: None,
                finish_reason: Some(map_finish_reason(&reason)),
                usage: None,
            });
        }
    }

    Ok(updates)
}

#[derive(Deserialize)]
struct CopilotStreamChunk {
    #[serde(default)]
    choices: Vec<CopilotStreamChoice>,
    #[serde(default)]
    usage: Option<CopilotUsage>,
}

#[derive(Deserialize)]
struct CopilotStreamChoice {
    delta: CopilotDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize, Default)]
struct CopilotDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<CopilotStreamToolCall>>,
}

#[derive(Deserialize)]
struct CopilotStreamToolCall {
    index: u32,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<CopilotStreamFunctionCall>,
}

#[derive(Deserialize)]
struct CopilotStreamFunctionCall {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/// Read an HTTP response body into memory with a hard byte cap.
///
/// Mirrors the bounded-read helper used by the other provider crates: we
/// reject anything whose declared `Content-Length` exceeds the cap, and we
/// accumulate streamed chunks against the same cap so a provider that omits
/// `Content-Length` cannot OOM the process.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::types::Content;

    fn test_client() -> GitHubCopilotChatClient {
        let config = GitHubCopilotConfig {
            github_token: SecretString::new("ghp_test_token_value"),
            model: "gpt-4o".to_string(),
            editor_version: DEFAULT_EDITOR_VERSION.to_string(),
            integration_id: DEFAULT_INTEGRATION_ID.to_string(),
            base_url: "https://example.invalid".to_string(),
            max_tokens: 256,
            request_timeout: Duration::from_secs(5),
            connect_timeout: Duration::from_secs(1),
        };
        GitHubCopilotChatClient::new(config).expect("client builds")
    }

    #[test]
    fn from_env_reads_variables() {
        // Preserve any existing values so a developer shell is not mutated.
        let prev_token = std::env::var("GITHUB_TOKEN").ok();
        let prev_model = std::env::var("GITHUB_COPILOT_MODEL").ok();

        std::env::set_var("GITHUB_TOKEN", "ghp_fromenv");
        std::env::set_var("GITHUB_COPILOT_MODEL", "claude-3.5-sonnet");

        let config = GitHubCopilotConfig::from_env().expect("env config");
        assert_eq!(config.github_token.expose(), "ghp_fromenv");
        assert_eq!(config.model, "claude-3.5-sonnet");
        assert_eq!(config.editor_version, DEFAULT_EDITOR_VERSION);
        assert_eq!(config.integration_id, DEFAULT_INTEGRATION_ID);

        // Restore environment.
        match prev_token {
            Some(v) => std::env::set_var("GITHUB_TOKEN", v),
            None => std::env::remove_var("GITHUB_TOKEN"),
        }
        match prev_model {
            Some(v) => std::env::set_var("GITHUB_COPILOT_MODEL", v),
            None => std::env::remove_var("GITHUB_COPILOT_MODEL"),
        }
    }

    #[test]
    fn request_body_includes_messages_and_tools() {
        use agent_framework_core::tools::ToolDefinition;

        let client = test_client();
        let options = ChatOptions {
            temperature: Some(0.2),
            tools: vec![ToolDefinition::no_params("ping", "Ping the server").unwrap()],
            ..Default::default()
        };
        let body = client.build_request_body(
            &[Message::system("you are helpful"), Message::user("hello")],
            Some(&options),
        );

        assert_eq!(body.model, "gpt-4o");
        assert_eq!(body.messages.len(), 2);
        assert_eq!(body.messages[0].role, "system");
        assert_eq!(body.messages[1].role, "user");
        assert_eq!(
            body.messages[1].content.as_ref().and_then(|v| v.as_str()),
            Some("hello")
        );
        let tools = body.tools.expect("tools present");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "ping");
        assert_eq!(body.temperature, Some(0.2));
        // Default stream flag is None on the sync request path.
        assert!(body.stream.is_none());
    }

    #[test]
    fn build_headers_sets_copilot_identifiers() {
        let client = test_client();
        let headers = client.build_headers().expect("headers built");

        assert_eq!(
            headers.get(CONTENT_TYPE).and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
        assert_eq!(
            headers.get("editor-version").and_then(|v| v.to_str().ok()),
            Some(DEFAULT_EDITOR_VERSION)
        );
        assert_eq!(
            headers.get("copilot-integration-id").and_then(|v| v.to_str().ok()),
            Some(DEFAULT_INTEGRATION_ID)
        );

        // Authorization must be present, flagged as sensitive, and Bearer-prefixed.
        let auth = headers.get(AUTHORIZATION).expect("auth header present");
        assert!(auth.is_sensitive(), "auth header should be marked sensitive");
        assert!(auth
            .to_str()
            .expect("auth is ASCII")
            .starts_with("Bearer "));
    }

    #[test]
    fn tool_role_message_uses_tool_call_id() {
        let msg = Message::tool_result("call_42", "done");
        let out = message_to_copilot(&msg);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, "tool");
        assert_eq!(out[0].tool_call_id.as_deref(), Some("call_42"));
    }

    #[test]
    fn assistant_tool_call_serializes() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                Content::text("thinking..."),
                Content::tool_call("call_1", "lookup", serde_json::json!({"q": "hi"})),
            ],
            name: None,
            metadata: HashMap::new(),
        };
        let out = message_to_copilot(&msg);
        assert_eq!(out.len(), 1);
        let calls = out[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "lookup");
        assert!(calls[0].function.arguments.contains("hi"));
    }
}
