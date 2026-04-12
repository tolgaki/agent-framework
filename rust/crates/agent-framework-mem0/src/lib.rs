// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework — Mem0 long-term memory
//!
//! A Mem0 backed [`ContextProvider`] and REST client for the Microsoft Agent
//! Framework. Mirrors the .NET `Microsoft.Agents.AI.Mem0` project.
//!
//! The provider stores user/assistant conversation messages as Mem0 memories
//! and retrieves semantically-relevant memories for subsequent agent runs,
//! injecting them as additional system instructions.
//!
//! # Security considerations
//!
//! - **External service trust.** This crate communicates with a remote Mem0
//!   deployment over HTTP. Always use HTTPS in production and configure a
//!   trusted `base_url`.
//! - **PII and sensitive data.** Conversation content is sent to Mem0 for
//!   storage. Ensure the Mem0 deployment is configured with appropriate data
//!   retention and access controls.
//! - **Indirect prompt injection.** Memories returned by Mem0 are injected
//!   into the LLM context. A compromised memory store can influence agent
//!   behaviour; memories are accepted as-is and are not validated.
//!
//! # Example
//!
//! ```rust,no_run
//! use agent_framework_mem0::{Mem0Client, Mem0Config};
//! use agent_framework_core::secret::SecretString;
//!
//! # async fn example() -> agent_framework_core::error::AgentResult<()> {
//! let config = Mem0Config {
//!     api_key: SecretString::new("mem0-token"),
//!     base_url: "https://api.mem0.ai/v1".to_string(),
//!     user_id: Some("user-42".to_string()),
//! };
//! let client = Mem0Client::new(config)?;
//! let memories = client.search_memories("favourite colour", "user-42", 5).await?;
//! for m in memories {
//!     println!("{}: {}", m.score, m.memory);
//! }
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use tracing::debug;

use agent_framework_core::context::{AIContext, ContextProvider};
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::middleware::{AgentMiddleware, AgentMiddlewareNext};
use agent_framework_core::secret::SecretString;
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::{AgentResponse, Message, Role};

/// Default base URL for the hosted Mem0 service.
pub const DEFAULT_BASE_URL: &str = "https://api.mem0.ai/v1";

/// Default number of memories retrieved by [`Mem0ContextProvider`].
pub const DEFAULT_MEMORY_LIMIT: usize = 5;

/// Default context prompt prefixed to the retrieved memories when they are
/// injected as system instructions. Mirrors the .NET provider's default.
pub const DEFAULT_CONTEXT_PROMPT: &str = "Relevant memories:";

/// Default overall HTTP request timeout.
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
/// Default TCP connect timeout.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Mem0 client.
///
/// The `api_key` is wrapped in [`SecretString`] so it is redacted from
/// `Debug` output and zeroized on drop.
#[derive(Debug, Clone)]
pub struct Mem0Config {
    /// Mem0 API key (redacted in `Debug` output).
    pub api_key: SecretString,

    /// Base URL for the Mem0 API. Defaults to [`DEFAULT_BASE_URL`].
    pub base_url: String,

    /// Optional default user id used by [`Mem0ContextProvider`] when no
    /// session-level override is supplied.
    pub user_id: Option<String>,
}

impl Mem0Config {
    /// Create a config from environment variables.
    ///
    /// Reads:
    /// - `MEM0_API_KEY` — required.
    /// - `MEM0_BASE_URL` — optional, defaults to [`DEFAULT_BASE_URL`].
    /// - `MEM0_USER_ID` — optional, no default.
    pub fn from_env() -> AgentResult<Self> {
        let api_key = std::env::var("MEM0_API_KEY")
            .map_err(|_| AgentError::InvalidRequest("MEM0_API_KEY environment variable is not set".to_string()))?;
        let base_url = std::env::var("MEM0_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
        let user_id = std::env::var("MEM0_USER_ID").ok().filter(|s| !s.is_empty());

        Ok(Self {
            api_key: SecretString::new(api_key),
            base_url,
            user_id,
        })
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// A memory record returned by Mem0.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Memory {
    /// Stable server-assigned memory identifier.
    #[serde(default)]
    pub id: String,

    /// The memory text.
    pub memory: String,

    /// Relevance score assigned by Mem0 (higher = more relevant).
    #[serde(default)]
    pub score: f32,

    /// Opaque metadata attached to the memory, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Shape of a single message sent to the Mem0 `/memories` endpoint.
#[derive(Debug, Clone, Serialize)]
struct Mem0Message<'a> {
    role: &'a str,
    content: String,
}

/// Request body for `POST /memories`.
#[derive(Debug, Clone, Serialize)]
struct AddMemoryRequest<'a> {
    messages: Vec<Mem0Message<'a>>,
    user_id: &'a str,
}

/// Request body for `POST /memories/search`.
#[derive(Debug, Clone, Serialize)]
struct SearchMemoryRequest<'a> {
    query: &'a str,
    user_id: &'a str,
    limit: usize,
}

/// Response envelope for `POST /memories/search`. Mem0 may return either a
/// bare list or an object of the form `{"results": [...]}`; we accept both.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum SearchResponse {
    Wrapped { results: Vec<Memory> },
    Bare(Vec<Memory>),
}

impl SearchResponse {
    fn into_results(self) -> Vec<Memory> {
        match self {
            Self::Wrapped { results } => results,
            Self::Bare(results) => results,
        }
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// REST client for the Mem0 API.
///
/// Cloning is cheap: the underlying `reqwest::Client` already uses
/// `Arc`-backed connection pooling.
pub struct Mem0Client {
    config: Mem0Config,
    http: reqwest::Client,
}

impl Mem0Client {
    /// Create a new Mem0 client.
    ///
    /// Returns an error if the underlying HTTP client fails to build (e.g.
    /// the TLS backend cannot be initialised).
    pub fn new(config: Mem0Config) -> AgentResult<Self> {
        // Disable redirects: reqwest's default policy strips `Authorization`
        // on cross-origin hops, but if we followed a redirect from a hostile
        // `base_url` we could still leak the token in the request body or to
        // an unexpected host. Be conservative and refuse to follow.
        let http = reqwest::Client::builder()
            .timeout(DEFAULT_REQUEST_TIMEOUT)
            .connect_timeout(DEFAULT_CONNECT_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| AgentError::InvalidRequest(format!("failed to build HTTP client: {e}")))?;
        Ok(Self { config, http })
    }

    /// Create a client using the Mem0 config loaded from environment variables.
    pub fn from_env() -> AgentResult<Self> {
        Self::new(Mem0Config::from_env()?)
    }

    /// Return the configured base URL.
    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    /// Return the configured default user id, if any.
    pub fn default_user_id(&self) -> Option<&str> {
        self.config.user_id.as_deref()
    }

    /// Build the `Authorization: Token ...` / `Content-Type` headers for a
    /// Mem0 API request.
    fn build_headers(&self) -> AgentResult<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let value = format!("Token {}", self.config.api_key.expose());
        let mut header = HeaderValue::from_str(&value)
            .map_err(|_| AgentError::InvalidRequest("MEM0_API_KEY contains invalid characters".to_string()))?;
        header.set_sensitive(true);
        headers.insert(AUTHORIZATION, header);
        Ok(headers)
    }

    /// Join the configured base URL with a relative path, handling a
    /// trailing slash on `base_url` gracefully.
    fn endpoint(&self, path: &str) -> String {
        let base = self.config.base_url.trim_end_matches('/');
        format!("{}/{}", base, path.trim_start_matches('/'))
    }

    /// Store a set of conversation messages as memories.
    ///
    /// Only messages whose role is user, assistant, or system are persisted;
    /// tool-call and tool-result messages are skipped. Messages with no text
    /// content are also skipped. Returns `Ok(())` if there is nothing to store.
    pub async fn add_memory(&self, messages: &[Message], user_id: &str) -> AgentResult<()> {
        let payload: Vec<Mem0Message<'_>> = messages
            .iter()
            .filter_map(|m| {
                let role = match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System => "system",
                    Role::Tool => return None,
                };
                let text = m.text();
                if text.trim().is_empty() {
                    None
                } else {
                    Some(Mem0Message { role, content: text })
                }
            })
            .collect();

        if payload.is_empty() {
            debug!("Mem0: add_memory called with no persistable messages; skipping");
            return Ok(());
        }

        let body = AddMemoryRequest {
            messages: payload,
            user_id,
        };
        let url = self.endpoint("memories");
        debug!(url = %url, user_id = %user_id, "Mem0: POST /memories");

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
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::provider(
                format!("Mem0 add_memory failed ({status}): {text}"),
                Some(status.as_u16()),
            ));
        }
        Ok(())
    }

    /// Build the search request body without actually sending it.
    ///
    /// Exposed for testing and for callers that want to inspect the wire
    /// format before dispatch.
    fn build_search_body<'a>(query: &'a str, user_id: &'a str, limit: usize) -> SearchMemoryRequest<'a> {
        SearchMemoryRequest { query, user_id, limit }
    }

    /// Search for memories relevant to `query`, scoped to `user_id`.
    pub async fn search_memories(&self, query: &str, user_id: &str, limit: usize) -> AgentResult<Vec<Memory>> {
        let body = Self::build_search_body(query, user_id, limit);
        let url = self.endpoint("memories/search");
        debug!(url = %url, user_id = %user_id, limit = limit, "Mem0: POST /memories/search");

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
            let text = response.text().await.unwrap_or_default();
            return Err(AgentError::provider(
                format!("Mem0 search_memories failed ({status}): {text}"),
                Some(status.as_u16()),
            ));
        }

        let bytes = response
            .bytes()
            .await
            .map_err(|e| AgentError::HttpError(format!("failed to read Mem0 response body: {e}")))?;
        let parsed: SearchResponse = serde_json::from_slice(&bytes)?;
        Ok(parsed.into_results())
    }
}

// ---------------------------------------------------------------------------
// ContextProvider
// ---------------------------------------------------------------------------

/// A [`ContextProvider`] that injects Mem0 memories as system instructions.
///
/// On [`provide_context`](ContextProvider::provide_context) the provider
/// searches Mem0 for memories relevant to the current session and, if any are
/// returned, prepends them as a single system-instruction block of the form:
///
/// ```text
/// Relevant memories:
/// - memory one
/// - memory two
/// ```
pub struct Mem0ContextProvider {
    client: Mem0Client,
    limit: usize,
    context_prompt: String,
    user_id_override: Option<String>,
}

impl Mem0ContextProvider {
    /// Create a new provider that fetches up to [`DEFAULT_MEMORY_LIMIT`]
    /// memories using [`DEFAULT_CONTEXT_PROMPT`] as the header line.
    pub fn new(client: Mem0Client) -> Self {
        Self {
            client,
            limit: DEFAULT_MEMORY_LIMIT,
            context_prompt: DEFAULT_CONTEXT_PROMPT.to_string(),
            user_id_override: None,
        }
    }

    /// Override the maximum number of memories fetched per run.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Override the prompt prepended to the injected memories.
    pub fn with_context_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.context_prompt = prompt.into();
        self
    }

    /// Override the Mem0 `user_id` used for every lookup.
    ///
    /// If not set, falls back to the `user_id` configured on the underlying
    /// client, then to the session id.
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id_override = Some(user_id.into());
        self
    }

    /// Resolve the `user_id` used for a given session.
    fn resolve_user_id<'a>(&'a self, session: &'a AgentSession) -> &'a str {
        if let Some(id) = &self.user_id_override {
            return id.as_str();
        }
        if let Some(id) = self.client.default_user_id() {
            return id;
        }
        session.session_id.as_str()
    }
}

#[async_trait]
impl ContextProvider for Mem0ContextProvider {
    async fn provide_context(&self, session: &AgentSession) -> AgentResult<AIContext> {
        let user_id = self.resolve_user_id(session);
        // Mem0's search requires a query string. In the absence of a pending
        // user message we use the session id as a stable bucket key — the
        // caller can wrap this provider to supply a more meaningful query.
        let query = session.session_id.as_str();

        let memories = match self.client.search_memories(query, user_id, self.limit).await {
            Ok(m) => m,
            Err(e) => {
                // Failures here must NOT abort the agent run — missing memories
                // degrade the response gracefully but do not corrupt it.
                debug!(error = %e, "Mem0: search_memories failed; returning empty context");
                return Ok(AIContext::default());
            }
        };

        if memories.is_empty() {
            return Ok(AIContext::default());
        }

        let mut buf = String::with_capacity(self.context_prompt.len() + 64 * memories.len());
        buf.push_str(&self.context_prompt);
        buf.push('\n');
        for m in &memories {
            buf.push_str("- ");
            buf.push_str(&m.memory);
            buf.push('\n');
        }

        Ok(AIContext {
            instructions: Some(buf),
            messages: Vec::new(),
            tools: Vec::new(),
        })
    }
}

// ---------------------------------------------------------------------------
// Agent middleware — persist run output to Mem0
// ---------------------------------------------------------------------------

/// An [`AgentMiddleware`] that, after a successful agent run, persists the
/// new conversation messages to Mem0 as long-term memories.
///
/// Failures while persisting are logged and swallowed so that storage errors
/// never mask a successful agent response.
pub struct Mem0Middleware {
    client: Mem0Client,
    user_id_override: Option<String>,
}

impl Mem0Middleware {
    /// Create a middleware that persists runs to Mem0 via `client`.
    pub fn new(client: Mem0Client) -> Self {
        Self {
            client,
            user_id_override: None,
        }
    }

    /// Override the Mem0 `user_id` used when persisting runs. Defaults to the
    /// client's configured user id, or the session id when neither is set.
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id_override = Some(user_id.into());
        self
    }

    fn resolve_user_id<'a>(&'a self, session: &'a AgentSession) -> &'a str {
        if let Some(id) = &self.user_id_override {
            return id.as_str();
        }
        if let Some(id) = self.client.default_user_id() {
            return id;
        }
        session.session_id.as_str()
    }
}

#[async_trait]
impl AgentMiddleware for Mem0Middleware {
    async fn on_run(
        &self,
        messages: Vec<Message>,
        session: &mut AgentSession,
        next: &dyn AgentMiddlewareNext,
    ) -> AgentResult<AgentResponse> {
        // The inbound messages belong to the new turn — persist them along
        // with the assistant reply so both ends of the exchange become
        // memories.
        let inbound = messages.clone();
        let response = next.run(messages, session).await?;

        let mut to_persist = inbound;
        to_persist.extend(response.messages.iter().cloned());

        let user_id = self.resolve_user_id(session).to_string();
        if let Err(e) = self.client.add_memory(&to_persist, &user_id).await {
            // Storage failures must NOT bubble up — the agent already produced
            // a valid response. Log and continue.
            debug!(error = %e, "Mem0: add_memory failed; continuing");
        }

        Ok(response)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Guard to keep env-var tests from racing with each other. Tests in the
    // same module share the process environment.
    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        use std::sync::{Mutex, OnceLock};
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn config_from_env_reads_all_three_variables() {
        let _g = env_guard();
        // SAFETY: guarded by the module-scoped mutex above.
        unsafe {
            std::env::set_var("MEM0_API_KEY", "token-abc");
            std::env::set_var("MEM0_BASE_URL", "https://example.invalid/v2");
            std::env::set_var("MEM0_USER_ID", "user-42");
        }

        let cfg = Mem0Config::from_env().expect("from_env should succeed");
        assert_eq!(cfg.api_key.expose(), "token-abc");
        assert_eq!(cfg.base_url, "https://example.invalid/v2");
        assert_eq!(cfg.user_id.as_deref(), Some("user-42"));

        // Clear and check that missing MEM0_API_KEY is an error, and that
        // MEM0_BASE_URL falls back to the default.
        unsafe {
            std::env::remove_var("MEM0_API_KEY");
            std::env::remove_var("MEM0_BASE_URL");
            std::env::remove_var("MEM0_USER_ID");
        }
        let err = Mem0Config::from_env().expect_err("missing key should fail");
        assert!(format!("{err}").contains("MEM0_API_KEY"));

        unsafe {
            std::env::set_var("MEM0_API_KEY", "k");
        }
        let cfg = Mem0Config::from_env().unwrap();
        assert_eq!(cfg.base_url, DEFAULT_BASE_URL);
        assert!(cfg.user_id.is_none());
        unsafe {
            std::env::remove_var("MEM0_API_KEY");
        }
    }

    #[test]
    fn memory_deserializes_from_both_shapes() {
        // Bare list.
        let bare = br#"[{"id":"m1","memory":"likes tea","score":0.91}]"#;
        let parsed: SearchResponse = serde_json::from_slice(bare).unwrap();
        let mems = parsed.into_results();
        assert_eq!(mems.len(), 1);
        assert_eq!(mems[0].id, "m1");
        assert_eq!(mems[0].memory, "likes tea");
        assert!((mems[0].score - 0.91).abs() < 1e-6);
        assert!(mems[0].metadata.is_none());

        // Wrapped object with metadata.
        let wrapped = br#"{"results":[{"id":"m2","memory":"loves rust","score":0.5,"metadata":{"source":"chat"}}]}"#;
        let parsed: SearchResponse = serde_json::from_slice(wrapped).unwrap();
        let mems = parsed.into_results();
        assert_eq!(mems.len(), 1);
        assert_eq!(mems[0].id, "m2");
        assert_eq!(mems[0].memory, "loves rust");
        assert_eq!(
            mems[0].metadata.as_ref().and_then(|v| v.get("source")).and_then(|v| v.as_str()),
            Some("chat")
        );
    }

    #[test]
    fn search_request_builds_correct_json() {
        let body = Mem0Client::build_search_body("favourite colour", "user-42", 7);
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["query"], "favourite colour");
        assert_eq!(json["user_id"], "user-42");
        assert_eq!(json["limit"], 7);

        // Endpoint join should handle both trailing and non-trailing slashes.
        let cfg = Mem0Config {
            api_key: SecretString::new("k"),
            base_url: "https://api.mem0.ai/v1/".to_string(),
            user_id: None,
        };
        let client = Mem0Client::new(cfg).unwrap();
        assert_eq!(client.endpoint("memories/search"), "https://api.mem0.ai/v1/memories/search");
        assert_eq!(client.endpoint("/memories"), "https://api.mem0.ai/v1/memories");
    }
}
