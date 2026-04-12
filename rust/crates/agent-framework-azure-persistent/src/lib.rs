// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework Azure Persistent Agents
//!
//! Azure AI Persistent Agents provider for the Microsoft Agent Framework.
//!
//! This crate mirrors the .NET `Microsoft.Agents.AI.AzureAI.Persistent`
//! project. Unlike [`agent-framework-foundry`], which talks to a stateless
//! chat completions endpoint, this crate integrates with the *persistent
//! agents* service: thread state (messages, runs) lives on the server and
//! is referenced by id.
//!
//! # Lifecycle
//!
//! For each [`PersistentAgent::run`] call:
//!
//! 1. If the [`AgentSession`] has no `conversation_id`, a new thread is
//!    created and the id is stored on the session so subsequent calls
//!    reuse the same server-side thread.
//! 2. Incoming user messages are appended to the thread.
//! 3. A run is created against the configured `agent_id` and polled
//!    until it reaches a terminal state.
//! 4. The newest assistant message is returned as an [`AgentResponse`].
//!
//! # Endpoints
//!
//! - `POST {endpoint}/threads`
//! - `POST {endpoint}/threads/{id}/messages`
//! - `POST {endpoint}/threads/{id}/runs`
//! - `GET  {endpoint}/threads/{id}/runs/{run_id}`
//! - `GET  {endpoint}/threads/{id}/messages`
//!
//! Auth is supplied via an `api-key` header.

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::Deserialize;
use tracing::debug;

use agent_framework_core::agent::Agent;
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::http_limits::DEFAULT_MAX_RESPONSE_BYTES;
use agent_framework_core::redact::{scrub_error_body, MAX_ERROR_BODY_LEN};
use agent_framework_core::secret::SecretString;
use agent_framework_core::session::AgentSession;
use agent_framework_core::streaming::AgentResponseStream;
use agent_framework_core::types::{
    AgentResponse, AgentRunOptions, Content, FinishReason, Message, Role,
};

const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Interval between `GET runs/{id}` polls while waiting for completion.
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(500);
/// Maximum time a single `run()` call will wait for a run to complete.
const DEFAULT_RUN_TIMEOUT: Duration = Duration::from_secs(300);

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for an Azure AI Persistent Agents client.
#[derive(Clone, Debug)]
pub struct PersistentConfig {
    /// Service endpoint, e.g. `https://my-project.services.ai.azure.com`.
    pub endpoint: String,
    /// API key (sent in the `api-key` header).
    pub api_key: SecretString,
    /// Identifier of the persistent agent to run.
    pub agent_id: String,
    /// Overall request timeout for individual HTTP calls.
    pub request_timeout: Duration,
    /// TCP connect timeout.
    pub connect_timeout: Duration,
    /// Interval between run-status polls.
    pub poll_interval: Duration,
    /// Maximum time to wait for a single run to reach a terminal state.
    pub run_timeout: Duration,
}

impl PersistentConfig {
    /// Build a config from environment variables.
    ///
    /// Reads:
    /// - `AZURE_AI_ENDPOINT` (required)
    /// - `AZURE_AI_API_KEY` (required)
    /// - `AZURE_AI_AGENT_ID` (required)
    pub fn from_env() -> AgentResult<Self> {
        let endpoint = std::env::var("AZURE_AI_ENDPOINT").map_err(|_| {
            AgentError::InvalidRequest("AZURE_AI_ENDPOINT environment variable is not set".to_string())
        })?;
        let api_key = std::env::var("AZURE_AI_API_KEY").map_err(|_| {
            AgentError::InvalidRequest("AZURE_AI_API_KEY environment variable is not set".to_string())
        })?;
        let agent_id = std::env::var("AZURE_AI_AGENT_ID").map_err(|_| {
            AgentError::InvalidRequest("AZURE_AI_AGENT_ID environment variable is not set".to_string())
        })?;

        Ok(Self {
            endpoint,
            api_key: SecretString::new(api_key),
            agent_id,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            poll_interval: DEFAULT_POLL_INTERVAL,
            run_timeout: DEFAULT_RUN_TIMEOUT,
        })
    }

    fn base(&self) -> &str {
        self.endpoint.trim_end_matches('/')
    }
}

// ---------------------------------------------------------------------------
// REST client
// ---------------------------------------------------------------------------

/// Low-level REST client for the Azure AI Persistent Agents API.
///
/// Use this directly for fine-grained control, or wrap it in a
/// [`PersistentAgent`] to get the framework's `Agent` trait semantics.
pub struct PersistentAgentsClient {
    config: PersistentConfig,
    http: reqwest::Client,
}

impl PersistentAgentsClient {
    /// Create a new client from a config.
    pub fn new(config: PersistentConfig) -> AgentResult<Self> {
        // `Policy::none()` is deliberate: we never want the api-key header
        // to follow a redirect to an attacker-controlled host.
        let http = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .connect_timeout(config.connect_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| AgentError::InvalidRequest(format!("failed to build HTTP client: {e}")))?;
        Ok(Self { config, http })
    }

    fn build_headers(&self) -> AgentResult<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let mut api_key = HeaderValue::from_str(self.config.api_key.expose())
            .map_err(|_| AgentError::InvalidRequest("AZURE_AI_API_KEY contains invalid characters".to_string()))?;
        api_key.set_sensitive(true);
        headers.insert("api-key", api_key);
        Ok(headers)
    }

    /// Send a request and decode a JSON response with a size cap.
    async fn send_json<T: for<'de> Deserialize<'de>>(
        &self,
        request: reqwest::RequestBuilder,
        context: &str,
    ) -> AgentResult<T> {
        let response = request
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
                    "Azure AI Persistent Agents error during {context} ({status}): {}",
                    scrub_error_body(&error_text)
                ),
                Some(status.as_u16()),
            ));
        }
        let bytes = read_bounded_body(response, DEFAULT_MAX_RESPONSE_BYTES).await?;
        serde_json::from_slice::<T>(&bytes).map_err(AgentError::from)
    }

    /// Create a new thread. Returns the thread id.
    pub async fn create_thread(&self) -> AgentResult<String> {
        let url = format!("{}/threads", self.config.base());
        debug!(%url, "Creating persistent thread");

        // Body is an empty object — the service accepts defaults.
        let request = self
            .http
            .post(&url)
            .headers(self.build_headers()?)
            .json(&serde_json::json!({}));

        let thread: ThreadResource = self.send_json(request, "create_thread").await?;
        Ok(thread.id)
    }

    /// Append a single message to a thread.
    pub async fn add_message(&self, thread_id: &str, role: &str, content: &str) -> AgentResult<()> {
        let url = format!("{}/threads/{}/messages", self.config.base(), thread_id);
        debug!(%thread_id, %role, "Adding message to thread");

        let body = serde_json::json!({
            "role": role,
            "content": content,
        });
        let request = self.http.post(&url).headers(self.build_headers()?).json(&body);
        // Discard the response body but still enforce size limits + error handling.
        let _: serde_json::Value = self.send_json(request, "add_message").await?;
        Ok(())
    }

    /// Create a run against this thread + agent and poll until it finishes,
    /// returning the newest assistant message's text.
    pub async fn run_agent(&self, thread_id: &str) -> AgentResult<String> {
        let url = format!("{}/threads/{}/runs", self.config.base(), thread_id);
        debug!(%thread_id, agent = %self.config.agent_id, "Starting run");

        let body = serde_json::json!({
            "assistant_id": self.config.agent_id,
        });
        let request = self.http.post(&url).headers(self.build_headers()?).json(&body);
        let run: RunResource = self.send_json(request, "create_run").await?;

        // Poll until the run reaches a terminal state or we time out.
        let started = std::time::Instant::now();
        let terminal = loop {
            if started.elapsed() > self.config.run_timeout {
                return Err(AgentError::provider(
                    format!(
                        "run {} exceeded timeout of {:?}",
                        run.id, self.config.run_timeout
                    ),
                    None,
                ));
            }

            tokio::time::sleep(self.config.poll_interval).await;

            let status_url = format!(
                "{}/threads/{}/runs/{}",
                self.config.base(),
                thread_id,
                run.id
            );
            let get = self.http.get(&status_url).headers(self.build_headers()?);
            let current: RunResource = self.send_json(get, "poll_run").await?;

            debug!(run = %current.id, status = %current.status, "Run status");
            match current.status.as_str() {
                "completed" => break current,
                "failed" | "cancelled" | "expired" => {
                    let reason = current
                        .last_error
                        .as_ref()
                        .map(|e| format!("{}: {}", e.code, e.message))
                        .unwrap_or_else(|| "no error details".to_string());
                    return Err(AgentError::provider(
                        format!("run {} ended with status '{}' ({reason})", current.id, current.status),
                        None,
                    ));
                }
                // In-flight states: keep polling.
                "queued" | "in_progress" | "requires_action" | "cancelling" => continue,
                // Unknown state: keep polling but note it — the service may add
                // new states over time and crashing is worse than waiting.
                other => {
                    debug!(status = %other, "unrecognised run status, continuing to poll");
                }
            }
        };

        // Fetch messages and return the newest assistant reply.
        let messages = self.get_messages(thread_id).await?;
        let assistant_text = messages
            .into_iter()
            .rev()
            .find(|m| m.role == Role::Assistant)
            .map(|m| m.text())
            .unwrap_or_default();

        debug!(run = %terminal.id, "Run completed");
        Ok(assistant_text)
    }

    /// List messages in a thread, converted to framework [`Message`]s.
    ///
    /// The Azure API returns messages newest-first; we reverse so the
    /// returned vec is in chronological (oldest-first) order.
    pub async fn get_messages(&self, thread_id: &str) -> AgentResult<Vec<Message>> {
        let url = format!("{}/threads/{}/messages", self.config.base(), thread_id);
        let request = self.http.get(&url).headers(self.build_headers()?);
        let list: MessageList = self.send_json(request, "list_messages").await?;

        let mut out: Vec<Message> = list.data.into_iter().map(persistent_to_message).collect();
        out.reverse();
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Framework Agent wrapper
// ---------------------------------------------------------------------------

/// A framework [`Agent`] backed by an Azure AI Persistent Agents thread.
///
/// The agent reuses a single server-side thread per [`AgentSession`]:
/// the thread id is stored in [`AgentSession::conversation_id`], so
/// consecutive `run()` calls on the same session will continue the
/// same conversation without replaying history locally.
pub struct PersistentAgent {
    id: String,
    name: Option<String>,
    description: Option<String>,
    client: PersistentAgentsClient,
}

impl PersistentAgent {
    /// Build a new [`PersistentAgent`] from a config.
    pub fn new(config: PersistentConfig) -> AgentResult<Self> {
        let client = PersistentAgentsClient::new(config)?;
        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: None,
            description: None,
            client,
        })
    }

    /// Set an optional human-readable name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set an optional description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Access the underlying REST client for advanced operations.
    pub fn client(&self) -> &PersistentAgentsClient {
        &self.client
    }

    /// Ensure the session has a thread id, creating one if necessary.
    async fn ensure_thread(&self, session: &mut AgentSession) -> AgentResult<String> {
        if let Some(id) = &session.conversation_id {
            return Ok(id.clone());
        }
        let id = self.client.create_thread().await?;
        debug!(thread = %id, "Created new persistent thread for session");
        session.conversation_id = Some(id.clone());
        Ok(id)
    }
}

#[async_trait]
impl Agent for PersistentAgent {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    async fn run(
        &self,
        messages: Vec<Message>,
        session: &mut AgentSession,
        _options: Option<&AgentRunOptions>,
    ) -> AgentResult<AgentResponse> {
        let thread_id = self.ensure_thread(session).await?;

        // Forward every incoming user / system message to the persistent
        // thread. Assistant and tool messages in the input are ignored
        // because the service manages them on the server side.
        for msg in &messages {
            let role = match msg.role {
                Role::User => "user",
                // Azure persistent agents don't accept a system role on
                // `POST /messages`; fold system messages into user content
                // so they still reach the model.
                Role::System => "user",
                Role::Assistant => continue,
                Role::Tool => continue,
            };
            let text = msg.text();
            if text.is_empty() {
                continue;
            }
            self.client.add_message(&thread_id, role, &text).await?;
        }

        let reply_text = self.client.run_agent(&thread_id).await?;

        let assistant_msg = Message::assistant(reply_text.clone());
        let mut all_messages = messages;
        all_messages.push(assistant_msg);

        // Persist locally as well so `session.get_history()` still works
        // for callers that don't distinguish between server/local history.
        session.save_history(&all_messages).await?;

        Ok(AgentResponse {
            messages: all_messages,
            text: reply_text,
            finish_reason: Some(FinishReason::Stop),
            usage: None,
        })
    }

    fn run_stream<'a>(
        &'a self,
        _messages: Vec<Message>,
        _session: &'a mut AgentSession,
        _options: Option<&'a AgentRunOptions>,
    ) -> AgentResult<AgentResponseStream<'a>> {
        // The persistent agents service exposes a server-sent events
        // stream, but wiring that up requires additional dependencies
        // (`futures`, `reqwest-eventsource`) that this crate deliberately
        // avoids. Surface a clear error so callers fall back to `run()`.
        Err(AgentError::Unimplemented(
            "streaming is not yet implemented for PersistentAgent".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ThreadResource {
    id: String,
}

#[derive(Deserialize)]
struct RunResource {
    id: String,
    status: String,
    #[serde(default)]
    last_error: Option<RunError>,
}

#[derive(Deserialize)]
struct RunError {
    code: String,
    message: String,
}

#[derive(Deserialize)]
struct MessageList {
    #[serde(default)]
    data: Vec<PersistentMessage>,
}

#[derive(Deserialize)]
struct PersistentMessage {
    #[allow(dead_code)]
    #[serde(default)]
    id: Option<String>,
    role: String,
    #[serde(default)]
    content: Vec<PersistentContent>,
}

/// The persistent agents API encodes message content as an array of
/// typed parts. We currently surface `text` parts and ignore others
/// (images, files) — tool parts aren't exposed on the thread list API.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum PersistentContent {
    Text { text: PersistentText },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
struct PersistentText {
    value: String,
}

fn persistent_to_message(msg: PersistentMessage) -> Message {
    let role = match msg.role.as_str() {
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "system" => Role::System,
        _ => Role::User,
    };
    let content: Vec<Content> = msg
        .content
        .into_iter()
        .filter_map(|c| match c {
            PersistentContent::Text { text } => Some(Content::text(text.value)),
            PersistentContent::Other => None,
        })
        .collect();
    Message {
        role,
        content,
        name: None,
        metadata: HashMap::new(),
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

    fn test_config() -> PersistentConfig {
        PersistentConfig {
            endpoint: "https://example.services.ai.azure.com".to_string(),
            api_key: SecretString::new("test-key"),
            agent_id: "asst_abc123".to_string(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            poll_interval: DEFAULT_POLL_INTERVAL,
            run_timeout: DEFAULT_RUN_TIMEOUT,
        }
    }

    #[test]
    fn from_env_reads_all_required_variables() {
        // SAFETY: these env variable names are unique to this test.
        std::env::set_var("AZURE_AI_ENDPOINT", "https://foo.services.ai.azure.com");
        std::env::set_var("AZURE_AI_API_KEY", "sk-persistent-test");
        std::env::set_var("AZURE_AI_AGENT_ID", "asst_xyz");

        let cfg = PersistentConfig::from_env().expect("env should parse");
        assert_eq!(cfg.endpoint, "https://foo.services.ai.azure.com");
        assert_eq!(cfg.api_key.expose(), "sk-persistent-test");
        assert_eq!(cfg.agent_id, "asst_xyz");

        let dbg = format!("{cfg:?}");
        assert!(!dbg.contains("sk-persistent-test"));
        assert!(dbg.contains("REDACTED"));

        std::env::remove_var("AZURE_AI_ENDPOINT");
        std::env::remove_var("AZURE_AI_API_KEY");
        std::env::remove_var("AZURE_AI_AGENT_ID");
    }

    #[test]
    fn client_builds_and_headers_are_correct() {
        let cfg = test_config();
        assert_eq!(cfg.base(), "https://example.services.ai.azure.com");
        let client = PersistentAgentsClient::new(cfg).expect("client builds");
        let headers = client.build_headers().expect("headers build");
        assert_eq!(headers["content-type"], "application/json");
        assert!(headers.contains_key("api-key"));
        assert!(headers["api-key"].is_sensitive());

        // The framework agent wrapper should build successfully with a
        // unique id and no name/description by default.
        let agent = PersistentAgent::new(test_config()).expect("agent builds");
        assert!(!agent.id().is_empty());
        assert!(agent.name().is_none());
        assert!(agent.description().is_none());
        let named = PersistentAgent::new(test_config())
            .unwrap()
            .with_name("support-bot")
            .with_description("Handles support tickets");
        assert_eq!(named.name(), Some("support-bot"));
        assert_eq!(named.description(), Some("Handles support tickets"));
    }

    #[test]
    fn persistent_messages_decode_into_framework_messages() {
        // Assistant reply with a single text part — the common case.
        let json = serde_json::json!({
            "data": [
                {
                    "id": "msg_2",
                    "role": "assistant",
                    "content": [
                        { "type": "text", "text": { "value": "hello back" } }
                    ]
                },
                {
                    "id": "msg_1",
                    "role": "user",
                    "content": [
                        { "type": "text", "text": { "value": "hi" } }
                    ]
                }
            ]
        });
        let list: MessageList = serde_json::from_value(json).expect("parse");
        let mut converted: Vec<Message> = list.data.into_iter().map(persistent_to_message).collect();
        converted.reverse();
        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].role, Role::User);
        assert_eq!(converted[0].text(), "hi");
        assert_eq!(converted[1].role, Role::Assistant);
        assert_eq!(converted[1].text(), "hello back");

        // Unknown content parts (e.g., images) should be silently skipped
        // rather than causing a decode failure.
        let json = serde_json::json!({
            "data": [
                {
                    "id": "msg_3",
                    "role": "assistant",
                    "content": [
                        { "type": "image_file", "image_file": { "file_id": "file_123" } },
                        { "type": "text", "text": { "value": "see above" } }
                    ]
                }
            ]
        });
        let list: MessageList = serde_json::from_value(json).expect("parse with unknown part");
        let mut converted: Vec<Message> = list.data.into_iter().map(persistent_to_message).collect();
        converted.reverse();
        assert_eq!(converted.len(), 1);
        assert_eq!(converted[0].text(), "see above");
    }
}
