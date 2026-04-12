// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework Foundry
//!
//! Azure AI Foundry provider for the Microsoft Agent Framework.
//!
//! This crate mirrors the .NET `Microsoft.Agents.AI.Foundry` project and
//! exposes [`FoundryChatClient`], a
//! [`ChatClient`](agent_framework_core::ChatClient) that talks to the
//! Azure AI Foundry chat completions endpoint.
//!
//! Foundry's wire format is OpenAI-compatible, but the URL layout and
//! authentication header differ:
//!
//! ```text
//! {endpoint}/openai/deployments/{model}/chat/completions?api-version=2024-10-01-preview
//! ```
//!
//! Auth is supplied as an `api-key` header (not `Authorization: Bearer`).
//!
//! # Quick start
//!
//! ```no_run
//! use agent_framework_foundry::{FoundryAgent, FoundryConfig};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let config = FoundryConfig::from_env()?;
//! let agent = FoundryAgent::builder(config)?
//!     .instructions("You are a helpful assistant.")
//!     .build()?;
//! # Ok(()) }
//! ```

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use tracing::debug;

use agent_framework_core::agent::{ChatClientAgent, ChatClientAgentBuilder};
use agent_framework_core::client::ChatClient;
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::http_limits::DEFAULT_MAX_RESPONSE_BYTES;
use agent_framework_core::redact::{scrub_error_body, MAX_ERROR_BODY_LEN};
use agent_framework_core::secret::SecretString;
use agent_framework_core::streaming::ResponseStream;
use agent_framework_core::types::{
    ChatOptions, ChatResponse, Content, FinishReason, Message, Role, Usage,
};

const DEFAULT_API_VERSION: &str = "2024-10-01-preview";
const DEFAULT_MAX_TOKENS: u32 = 4096;
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for an Azure AI Foundry chat client.
///
/// The `api_key` field is wrapped in [`SecretString`] so it is never
/// printed by `Debug` and is zeroized when the config is dropped.
#[derive(Clone, Debug)]
pub struct FoundryConfig {
    /// Foundry resource endpoint, e.g. `https://myfoundry.services.ai.azure.com`.
    pub endpoint: String,
    /// API key (sent in the `api-key` header).
    pub api_key: SecretString,
    /// Foundry project / hub name.
    pub project_name: String,
    /// Deployed model name (acts as the `deployment` in the URL).
    pub model: String,
    /// API version to use. Defaults to `2024-10-01-preview`.
    pub api_version: String,
    /// Default maximum tokens to request per completion.
    pub max_tokens: u32,
    /// Overall request timeout.
    pub request_timeout: Duration,
    /// TCP connect timeout.
    pub connect_timeout: Duration,
}

impl FoundryConfig {
    /// Build a config from environment variables.
    ///
    /// Reads:
    /// - `FOUNDRY_ENDPOINT` (required)
    /// - `FOUNDRY_API_KEY` (required)
    /// - `FOUNDRY_PROJECT` (required)
    /// - `FOUNDRY_MODEL` (required)
    pub fn from_env() -> AgentResult<Self> {
        let endpoint = std::env::var("FOUNDRY_ENDPOINT").map_err(|_| {
            AgentError::InvalidRequest("FOUNDRY_ENDPOINT environment variable is not set".to_string())
        })?;
        let api_key = std::env::var("FOUNDRY_API_KEY").map_err(|_| {
            AgentError::InvalidRequest("FOUNDRY_API_KEY environment variable is not set".to_string())
        })?;
        let project_name = std::env::var("FOUNDRY_PROJECT").map_err(|_| {
            AgentError::InvalidRequest("FOUNDRY_PROJECT environment variable is not set".to_string())
        })?;
        let model = std::env::var("FOUNDRY_MODEL").map_err(|_| {
            AgentError::InvalidRequest("FOUNDRY_MODEL environment variable is not set".to_string())
        })?;

        Ok(Self {
            endpoint,
            api_key: SecretString::new(api_key),
            project_name,
            model,
            api_version: DEFAULT_API_VERSION.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        })
    }

    /// Construct the chat completions URL for the configured model.
    fn completions_url(&self) -> String {
        let base = self.endpoint.trim_end_matches('/');
        format!(
            "{base}/openai/deployments/{}/chat/completions?api-version={}",
            self.model, self.api_version
        )
    }
}

// ---------------------------------------------------------------------------
// Chat client
// ---------------------------------------------------------------------------

/// A [`ChatClient`] for Azure AI Foundry.
///
/// Uses the OpenAI-compatible chat completions wire format, but
/// authenticates via an `api-key` header and routes through the
/// Foundry deployment URL layout.
pub struct FoundryChatClient {
    config: FoundryConfig,
    http: reqwest::Client,
}

impl FoundryChatClient {
    /// Build a new chat client.
    ///
    /// Returns an error if the underlying reqwest client cannot be
    /// constructed (e.g. TLS backend failure).
    pub fn new(config: FoundryConfig) -> AgentResult<Self> {
        // `Policy::none()` is deliberate: reqwest's default redirect policy
        // strips `Authorization` / `Cookie` on cross-origin hops, but a
        // misconfigured `endpoint` could still lead the api-key header to
        // an attacker-controlled host via a same-origin bounce. API calls
        // should never follow redirects.
        let http = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .connect_timeout(config.connect_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| AgentError::InvalidRequest(format!("failed to build HTTP client: {e}")))?;
        Ok(Self { config, http })
    }

    /// Build the outbound headers (`Content-Type` + `api-key`).
    fn build_headers(&self) -> AgentResult<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let mut api_key = HeaderValue::from_str(self.config.api_key.expose())
            .map_err(|_| AgentError::InvalidRequest("FOUNDRY_API_KEY contains invalid characters".to_string()))?;
        api_key.set_sensitive(true);
        headers.insert("api-key", api_key);
        // The Foundry project is sent as an informational header so that
        // observability tooling can attribute requests. It is NOT required
        // by the API but mirrors how the .NET client tags calls.
        if let Ok(project) = HeaderValue::from_str(&self.config.project_name) {
            headers.insert("x-ms-foundry-project", project);
        }

        Ok(headers)
    }

    /// Convert framework [`Message`]s and [`ChatOptions`] into the OpenAI wire body.
    fn build_request_body(&self, messages: &[Message], options: Option<&ChatOptions>) -> OpenAIRequest {
        let max_tokens = options.and_then(|o| o.max_tokens).unwrap_or(self.config.max_tokens);

        let api_messages: Vec<OpenAIMessage> = messages.iter().flat_map(message_to_openai).collect();

        // Foundry uses OpenAI-compatible tool definitions. Include them when
        // they are present on the request options.
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
            // Foundry puts the deployment in the URL, but including the
            // model in the body is also accepted and useful for logs.
            model: options
                .and_then(|o| o.model.clone())
                .unwrap_or_else(|| self.config.model.clone()),
            messages: api_messages,
            max_tokens: Some(max_tokens),
            temperature: options.and_then(|o| o.temperature),
            top_p: options.and_then(|o| o.top_p),
            seed: options.and_then(|o| o.seed),
            frequency_penalty: options.and_then(|o| o.frequency_penalty),
            presence_penalty: options.and_then(|o| o.presence_penalty),
            stop: options.map(|o| o.stop_sequences.clone()).filter(|s| !s.is_empty()),
            user: options.and_then(|o| o.user.clone()),
            tools,
        }
    }
}

#[async_trait]
impl ChatClient for FoundryChatClient {
    async fn get_response(&self, messages: &[Message], options: Option<&ChatOptions>) -> AgentResult<ChatResponse> {
        let body = self.build_request_body(messages, options);
        let url = self.config.completions_url();

        debug!(model = %body.model, project = %self.config.project_name, "Sending request to Azure AI Foundry");

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
                format!("Foundry API error ({status}): {}", scrub_error_body(&error_text)),
                Some(status.as_u16()),
            ));
        }

        let bytes = read_bounded_body(response, DEFAULT_MAX_RESPONSE_BYTES).await?;
        let api_response: OpenAIResponse = serde_json::from_slice(&bytes)?;
        Ok(openai_response_to_chat_response(api_response))
    }

    async fn get_response_stream(
        &self,
        _messages: &[Message],
        _options: Option<&ChatOptions>,
    ) -> AgentResult<ResponseStream> {
        // Streaming parity with the OpenAI provider is tracked separately;
        // the baseline Foundry client returns a clear error so callers can
        // fall back to `get_response` without silent partial behaviour.
        Err(AgentError::Unimplemented(
            "streaming is not yet implemented for the Foundry provider".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Convenience agent wrapper
// ---------------------------------------------------------------------------

/// Convenience wrapper around [`ChatClientAgent`] pre-wired with a
/// [`FoundryChatClient`].
///
/// This mirrors the ergonomic `FoundryAgent` helper exposed by the .NET
/// `Microsoft.Agents.AI.Foundry` package. Use [`FoundryAgent::builder`]
/// to add instructions, tools, or middleware, or call
/// [`FoundryAgent::new`] for a minimal default agent.
pub struct FoundryAgent;

impl FoundryAgent {
    /// Build a default [`ChatClientAgent`] for the given config.
    ///
    /// This is a thin helper that builds a [`FoundryChatClient`], wraps it
    /// in a [`ChatClientAgent`] with framework defaults, and returns it.
    /// For non-default configuration (instructions, tools, middleware)
    /// use [`FoundryAgent::builder`] instead.
    pub fn create(config: FoundryConfig) -> AgentResult<ChatClientAgent> {
        Self::builder(config)?.build()
    }

    /// Start a [`ChatClientAgentBuilder`] pre-wired with a
    /// [`FoundryChatClient`]. Callers chain additional configuration
    /// (`instructions`, `tools`, etc.) and finish with `.build()`.
    pub fn builder(config: FoundryConfig) -> AgentResult<ChatClientAgentBuilder> {
        let client = FoundryChatClient::new(config)?;
        Ok(ChatClientAgent::builder().client(client))
    }
}

// ---------------------------------------------------------------------------
// OpenAI-compatible wire types (inlined to avoid depending on the openai crate)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
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
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
}

#[derive(Serialize)]
struct OpenAIMessage {
    role: String,
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
    #[serde(default)]
    id: Option<String>,
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
    #[serde(default)]
    role: Option<String>,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn message_to_openai(msg: &Message) -> Vec<OpenAIMessage> {
    let role = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };

    // Tool-role messages: one OpenAI message per ToolResult, tagged with
    // the corresponding tool_call_id. This matches the Foundry API.
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
        return vec![OpenAIMessage {
            role: "tool".to_string(),
            content: None,
            tool_calls: None,
            tool_call_id: None,
        }];
    }

    // Assistant tool calls get serialized alongside any text content.
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

    let text = msg.text();
    let content = if text.is_empty() {
        None
    } else {
        Some(serde_json::Value::String(text))
    };

    vec![OpenAIMessage {
        role: role.to_string(),
        content,
        tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
        tool_call_id: None,
    }]
}

fn openai_response_to_chat_response(resp: OpenAIResponse) -> ChatResponse {
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
        content_items.push(Content::text(text));
    }
    if let Some(tool_calls) = message.tool_calls {
        for tc in tool_calls {
            // Preserve the raw argument string if the model returns invalid
            // JSON, rather than silently becoming Null.
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
        response_id: resp.id,
        finish_reason,
        usage,
    }
}

/// Read an HTTP response body into memory with a hard byte cap.
///
/// Uses `reqwest::Response::chunk` (always available) rather than
/// `bytes_stream`, which would require the `futures` crate.
async fn read_bounded_body(mut response: reqwest::Response, max_bytes: usize) -> AgentResult<Vec<u8>> {
    if let Some(len) = response.content_length() {
        if len as u128 > max_bytes as u128 {
            return Err(AgentError::HttpError(format!(
                "response Content-Length {len} exceeds limit {max_bytes}"
            )));
        }
    }
    let mut buf = Vec::new();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                if buf.len().saturating_add(chunk.len()) > max_bytes {
                    return Err(AgentError::HttpError(format!(
                        "response body exceeded {max_bytes} bytes"
                    )));
                }
                buf.extend_from_slice(&chunk);
            }
            Ok(None) => break,
            Err(e) => return Err(AgentError::HttpError(e.to_string())),
        }
    }
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a config without touching the environment.
    fn test_config() -> FoundryConfig {
        FoundryConfig {
            endpoint: "https://example.services.ai.azure.com".to_string(),
            api_key: SecretString::new("test-key"),
            project_name: "my-project".to_string(),
            model: "gpt-4o".to_string(),
            api_version: DEFAULT_API_VERSION.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }

    #[test]
    fn from_env_reads_all_required_variables() {
        // Use unique variable names so this test does not collide with the
        // real environment or with other tests running in parallel.
        // SAFETY: std::env::set_var is unsafe in edition 2024 but this test
        // sets its own variables; no other test reads them.
        std::env::set_var("FOUNDRY_ENDPOINT", "https://foo.services.ai.azure.com");
        std::env::set_var("FOUNDRY_API_KEY", "sk-foundry-test");
        std::env::set_var("FOUNDRY_PROJECT", "proj-x");
        std::env::set_var("FOUNDRY_MODEL", "gpt-4o-mini");

        let cfg = FoundryConfig::from_env().expect("env should parse");
        assert_eq!(cfg.endpoint, "https://foo.services.ai.azure.com");
        assert_eq!(cfg.api_key.expose(), "sk-foundry-test");
        assert_eq!(cfg.project_name, "proj-x");
        assert_eq!(cfg.model, "gpt-4o-mini");
        assert_eq!(cfg.api_version, DEFAULT_API_VERSION);

        // Debug output must not leak the api_key.
        let dbg = format!("{cfg:?}");
        assert!(!dbg.contains("sk-foundry-test"));
        assert!(dbg.contains("REDACTED"));

        std::env::remove_var("FOUNDRY_ENDPOINT");
        std::env::remove_var("FOUNDRY_API_KEY");
        std::env::remove_var("FOUNDRY_PROJECT");
        std::env::remove_var("FOUNDRY_MODEL");
    }

    #[test]
    fn client_builds_and_targets_expected_url() {
        let cfg = test_config();
        let expected = "https://example.services.ai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-01-preview";
        assert_eq!(cfg.completions_url(), expected);
        let client = FoundryChatClient::new(cfg).expect("client builds");
        // Build headers and verify the api-key header is populated and marked
        // sensitive (so tracing middleware will not print it).
        let headers = client.build_headers().expect("headers build");
        assert!(headers.contains_key("api-key"));
        assert_eq!(headers["content-type"], "application/json");
        assert!(headers["api-key"].is_sensitive());
    }

    #[test]
    fn config_defaults_and_request_body_round_trip() {
        let cfg = test_config();
        assert_eq!(cfg.api_version, "2024-10-01-preview");
        assert_eq!(cfg.max_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(cfg.request_timeout, DEFAULT_REQUEST_TIMEOUT);
        assert_eq!(cfg.connect_timeout, DEFAULT_CONNECT_TIMEOUT);

        let client = FoundryChatClient::new(cfg).expect("client builds");
        let body = client.build_request_body(&[Message::user("hello")], None);
        assert_eq!(body.model, "gpt-4o");
        assert_eq!(body.max_tokens, Some(DEFAULT_MAX_TOKENS));
        assert_eq!(body.messages.len(), 1);
        assert_eq!(body.messages[0].role, "user");
        assert_eq!(
            body.messages[0].content.as_ref().and_then(|v| v.as_str()),
            Some("hello")
        );

        // Round-trip a sample provider response to confirm the parser
        // extracts text, finish reason, and usage fields.
        let sample = serde_json::json!({
            "id": "chatcmpl-1",
            "choices": [{
                "message": { "role": "assistant", "content": "hi there" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 5, "completion_tokens": 2 }
        });
        let parsed: OpenAIResponse = serde_json::from_value(sample).unwrap();
        let cr = openai_response_to_chat_response(parsed);
        assert_eq!(cr.response_id.as_deref(), Some("chatcmpl-1"));
        assert_eq!(cr.messages.len(), 1);
        assert_eq!(cr.messages[0].text(), "hi there");
        assert_eq!(cr.finish_reason, Some(FinishReason::Stop));
        let usage = cr.usage.expect("usage present");
        assert_eq!(usage.input_tokens, 5);
        assert_eq!(usage.output_tokens, 2);
    }
}
