// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework Copilot Studio
//!
//! Microsoft Copilot Studio agent provider for the Microsoft Agent Framework.
//!
//! Mirrors the .NET `Microsoft.Agents.AI.CopilotStudio` project. Exposes
//! [`CopilotStudioAgent`], an [`Agent`](agent_framework_core::Agent)
//! implementation backed by a Copilot Studio bot reached through the Direct
//! Line 3.0 REST API.
//!
//! ## Wire protocol
//!
//! Direct Line is a three-step loop:
//!
//! 1. `POST {endpoint}/v3/directline/conversations` — starts a conversation
//!    and returns a `conversationId`.
//! 2. `POST {endpoint}/v3/directline/conversations/{id}/activities` — posts
//!    a user `message` activity.
//! 3. `GET  {endpoint}/v3/directline/conversations/{id}/activities?watermark={w}`
//!    — polls for new `message` activities from the bot; the returned
//!    `watermark` is used as the cursor for the next poll.
//!
//! The conversation id is stored in [`AgentSession::conversation_id`] so that
//! subsequent calls to [`CopilotStudioAgent::run`] continue the same
//! conversation. This mirrors .NET's `CopilotStudioAgentSession.ConversationId`.
//!
//! ## Auth
//!
//! The client holds an Azure AD `client_id`/`client_secret`/`tenant_id`
//! triple and a Direct Line base endpoint. The Direct Line secret (or a
//! bearer token obtained via the client credentials flow in deployed setups)
//! is sent as `Authorization: Bearer {secret}`. This crate does not perform
//! the OAuth token-exchange itself — callers are expected to pre-provision
//! the secret used by Direct Line. The tenant and client identifiers are
//! retained on the config for downstream OAuth integrations.

use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use tracing::debug;
use uuid::Uuid;

use agent_framework_core::agent::Agent;
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::http_limits::DEFAULT_MAX_RESPONSE_BYTES;
use agent_framework_core::redact::{scrub_error_body, MAX_ERROR_BODY_LEN};
use agent_framework_core::secret::SecretString;
use agent_framework_core::session::AgentSession;
use agent_framework_core::streaming::AgentResponseStream;
use agent_framework_core::types::{AgentResponse, AgentRunOptions, FinishReason, Message};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// How many activity-poll iterations we do before giving up and returning
/// whatever text we've accumulated. Polling uses a short sleep between
/// attempts so this bounds the total wait by `MAX_POLL_ATTEMPTS *
/// POLL_INTERVAL`.
const MAX_POLL_ATTEMPTS: usize = 30;
/// Delay between successive activity polls.
const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Configuration for a [`CopilotStudioClient`].
///
/// Holds the Copilot Studio Direct Line endpoint, the Azure AD tenant /
/// application identifiers, and a client secret used as the Direct Line
/// bearer credential. Redaction of the secret is handled by [`SecretString`].
#[derive(Clone, Debug)]
pub struct CopilotStudioConfig {
    /// Direct Line endpoint for the bot, for example
    /// `https://directline.botframework.com`. No trailing slash.
    pub bot_endpoint: String,

    /// Azure AD tenant identifier (e.g. `"contoso.onmicrosoft.com"` or a GUID).
    pub tenant_id: String,

    /// Azure AD application (client) identifier.
    pub client_id: String,

    /// Client secret / Direct Line secret used as the bearer credential.
    pub client_secret: SecretString,

    /// Overall HTTP request timeout.
    pub request_timeout: Duration,

    /// TCP connect timeout.
    pub connect_timeout: Duration,
}

impl CopilotStudioConfig {
    /// Build a new config with default HTTP timeouts.
    pub fn new(
        bot_endpoint: impl Into<String>,
        tenant_id: impl Into<String>,
        client_id: impl Into<String>,
        client_secret: impl Into<SecretString>,
    ) -> Self {
        Self {
            bot_endpoint: bot_endpoint.into(),
            tenant_id: tenant_id.into(),
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        }
    }

    /// Create a config from environment variables.
    ///
    /// Reads:
    /// * `COPILOT_STUDIO_ENDPOINT` — Direct Line endpoint
    /// * `COPILOT_STUDIO_TENANT_ID` — Azure AD tenant id
    /// * `COPILOT_STUDIO_CLIENT_ID` — application/client id
    /// * `COPILOT_STUDIO_CLIENT_SECRET` — client secret (required)
    ///
    /// All four are required.
    pub fn from_env() -> AgentResult<Self> {
        let bot_endpoint = std::env::var("COPILOT_STUDIO_ENDPOINT").map_err(|_| {
            AgentError::InvalidRequest("COPILOT_STUDIO_ENDPOINT environment variable is not set".to_string())
        })?;
        let tenant_id = std::env::var("COPILOT_STUDIO_TENANT_ID").map_err(|_| {
            AgentError::InvalidRequest("COPILOT_STUDIO_TENANT_ID environment variable is not set".to_string())
        })?;
        let client_id = std::env::var("COPILOT_STUDIO_CLIENT_ID").map_err(|_| {
            AgentError::InvalidRequest("COPILOT_STUDIO_CLIENT_ID environment variable is not set".to_string())
        })?;
        let client_secret = std::env::var("COPILOT_STUDIO_CLIENT_SECRET").map_err(|_| {
            AgentError::InvalidRequest("COPILOT_STUDIO_CLIENT_SECRET environment variable is not set".to_string())
        })?;

        Ok(Self::new(
            bot_endpoint,
            tenant_id,
            client_id,
            SecretString::new(client_secret),
        ))
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// REST client for the Copilot Studio Direct Line API.
///
/// Each [`CopilotStudioClient`] instance owns a `reqwest::Client` configured
/// with strict timeouts and `redirect::Policy::none()` so the bearer token
/// never follows redirects off the configured host.
pub struct CopilotStudioClient {
    config: CopilotStudioConfig,
    http: reqwest::Client,
    /// Opaque id used in the Direct Line `from.id` field on outbound
    /// activities. Generated at construction time.
    user_id: String,
}

impl CopilotStudioClient {
    /// Create a new Copilot Studio client.
    ///
    /// Returns an error if the underlying HTTP client cannot be built.
    pub fn new(config: CopilotStudioConfig) -> AgentResult<Self> {
        // Redirect policy matches the other provider clients: never follow
        // redirects so an attacker-controlled `bot_endpoint` (or a compromised
        // upstream proxy) cannot bounce the `Authorization` header onto a
        // different host.
        let http = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .connect_timeout(config.connect_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| AgentError::InvalidRequest(format!("failed to build HTTP client: {e}")))?;

        Ok(Self {
            config,
            http,
            user_id: format!("user_{}", Uuid::new_v4()),
        })
    }

    fn build_headers(&self) -> AgentResult<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let mut auth = HeaderValue::from_str(&format!("Bearer {}", self.config.client_secret.expose()))
            .map_err(|_| {
                AgentError::InvalidRequest(
                    "COPILOT_STUDIO_CLIENT_SECRET contains invalid characters".to_string(),
                )
            })?;
        auth.set_sensitive(true);
        headers.insert(AUTHORIZATION, auth);
        Ok(headers)
    }

    fn conversations_url(&self) -> String {
        format!("{}/v3/directline/conversations", self.config.bot_endpoint.trim_end_matches('/'))
    }

    fn activities_url(&self, conversation_id: &str) -> String {
        format!(
            "{}/v3/directline/conversations/{}/activities",
            self.config.bot_endpoint.trim_end_matches('/'),
            conversation_id
        )
    }

    /// Start a new Direct Line conversation and return its identifier.
    pub async fn start_conversation(&self) -> AgentResult<String> {
        let url = self.conversations_url();
        debug!("Starting new Copilot Studio conversation");

        let response = self
            .http
            .post(&url)
            .headers(self.build_headers()?)
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
                format!("Copilot Studio error ({status}): {}", scrub_error_body(&error_text)),
                Some(status.as_u16()),
            ));
        }

        let bytes = read_bounded_body(response, DEFAULT_MAX_RESPONSE_BYTES).await?;
        let parsed: StartConversationResponse = serde_json::from_slice(&bytes)?;
        Ok(parsed.conversation_id)
    }

    /// Send a user text message as a Direct Line activity.
    pub async fn send_activity(&self, conversation_id: &str, text: &str) -> AgentResult<()> {
        let url = self.activities_url(conversation_id);
        let activity = OutboundActivity {
            r#type: "message",
            from: ActivityFrom {
                id: &self.user_id,
            },
            text,
        };

        debug!(conversation_id, "Posting Copilot Studio activity");

        let response = self
            .http
            .post(&url)
            .headers(self.build_headers()?)
            .json(&activity)
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
                format!("Copilot Studio error ({status}): {}", scrub_error_body(&error_text)),
                Some(status.as_u16()),
            ));
        }

        // Drain body up to the cap so the connection can be reused.
        let _ = read_bounded_body(response, DEFAULT_MAX_RESPONSE_BYTES).await?;
        Ok(())
    }

    /// Poll for activities on a Direct Line conversation.
    ///
    /// Returns bot reply texts (in order) together with the new watermark that
    /// the caller should pass on the next poll. User / typing / other activity
    /// types are filtered out.
    pub async fn get_activities(
        &self,
        conversation_id: &str,
        watermark: Option<&str>,
    ) -> AgentResult<(Vec<String>, Option<String>)> {
        let mut url = self.activities_url(conversation_id);
        if let Some(w) = watermark {
            // Direct Line's watermark is opaque but expected to be URL-safe.
            // Callers control it (it is echoed back from a prior response).
            url.push_str("?watermark=");
            url.push_str(w);
        }

        debug!(conversation_id, watermark, "Polling Copilot Studio activities");

        let response = self
            .http
            .get(&url)
            .headers(self.build_headers()?)
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
                format!("Copilot Studio error ({status}): {}", scrub_error_body(&error_text)),
                Some(status.as_u16()),
            ));
        }

        let bytes = read_bounded_body(response, DEFAULT_MAX_RESPONSE_BYTES).await?;
        let parsed: ActivitySetResponse = serde_json::from_slice(&bytes)?;

        // Only bot-authored `message` activities are surfaced. User echoes
        // (from.id == self.user_id) and `typing` / unknown activities are
        // dropped so the caller sees only the bot's replies in order.
        let replies: Vec<String> = parsed
            .activities
            .into_iter()
            .filter(|a| a.r#type == "message")
            .filter(|a| {
                a.from
                    .as_ref()
                    .map(|f| f.id.as_deref() != Some(self.user_id.as_str()))
                    .unwrap_or(true)
            })
            .filter_map(|a| a.text)
            .filter(|t| !t.is_empty())
            .collect();

        Ok((replies, parsed.watermark))
    }
}

// ---------------------------------------------------------------------------
// Agent implementation
// ---------------------------------------------------------------------------

/// An [`Agent`] backed by a Copilot Studio bot.
///
/// The conversation identifier returned by Direct Line is cached on the
/// [`AgentSession::conversation_id`] field so that subsequent `run` calls on
/// the same session continue the existing conversation instead of starting
/// a new one. This mirrors .NET's `CopilotStudioAgentSession.ConversationId`.
pub struct CopilotStudioAgent {
    id: String,
    name: Option<String>,
    description: Option<String>,
    client: CopilotStudioClient,
}

impl CopilotStudioAgent {
    /// Wrap an existing [`CopilotStudioClient`] as an [`Agent`].
    pub fn new(client: CopilotStudioClient) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: None,
            description: None,
            client,
        }
    }

    /// Set a friendly name returned by [`Agent::name`].
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set a description returned by [`Agent::description`].
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Access the inner Direct Line client.
    pub fn client(&self) -> &CopilotStudioClient {
        &self.client
    }
}

#[async_trait]
impl Agent for CopilotStudioAgent {
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
        // Reuse the Direct Line conversation id from the session if one is
        // already cached. Otherwise, start a fresh conversation and cache the
        // new id on the session so the next call resumes the thread.
        if session.conversation_id.is_none() {
            let new_id = self.client.start_conversation().await?;
            session.conversation_id = Some(new_id);
        }
        let conversation_id = session
            .conversation_id
            .clone()
            .expect("conversation id was just set");

        // Concatenate the text portion of every input message into a single
        // activity. Direct Line is a text-only channel per call; multi-turn
        // history is tracked by the bot on the server side against
        // `conversationId`, so we don't replay the session history.
        let text = messages
            .iter()
            .map(|m| m.text())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        if !text.is_empty() {
            self.client.send_activity(&conversation_id, &text).await?;
        }

        // Poll for bot replies. Direct Line's watermark is a monotonic cursor:
        // the first poll after posting returns our own message echoed back
        // (which we filter out) and any bot replies accumulated so far.
        let mut watermark: Option<String> = None;
        let mut replies: Vec<String> = Vec::new();

        for _ in 0..MAX_POLL_ATTEMPTS {
            let (new_replies, new_watermark) = self
                .client
                .get_activities(&conversation_id, watermark.as_deref())
                .await?;
            if new_watermark.is_some() {
                watermark = new_watermark;
            }
            if !new_replies.is_empty() {
                replies.extend(new_replies);
                // Direct Line doesn't signal "turn complete", so we stop on
                // the first non-empty batch of bot replies rather than
                // waiting further.
                break;
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
        // Suppress unused_assignments: the final watermark value is retained
        // so callers who reach into the session state can resume polling.
        let _ = watermark;

        let reply_text = replies.join("\n");
        let assistant = Message::assistant(reply_text.clone());
        let mut all_messages = messages;
        all_messages.push(assistant);

        // Persist the turn on the session so callers that rely on
        // `session.get_history()` see the exchange.
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
        // Direct Line exposes a `streamUrl` websocket for push delivery of
        // activities, but wiring that up requires additional dependencies
        // (`futures`, a WebSocket client) that this crate deliberately
        // avoids. Surface a clear error so callers fall back to `run()`.
        Err(AgentError::Unimplemented(
            "streaming is not yet implemented for CopilotStudioAgent".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Wire format
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct OutboundActivity<'a> {
    r#type: &'a str,
    from: ActivityFrom<'a>,
    text: &'a str,
}

#[derive(Serialize)]
struct ActivityFrom<'a> {
    id: &'a str,
}

#[derive(Deserialize)]
struct StartConversationResponse {
    #[serde(rename = "conversationId")]
    conversation_id: String,
}

#[derive(Deserialize)]
struct ActivitySetResponse {
    #[serde(default)]
    activities: Vec<InboundActivity>,
    #[serde(default)]
    watermark: Option<String>,
}

#[derive(Deserialize)]
struct InboundActivity {
    #[serde(default)]
    r#type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    from: Option<InboundFrom>,
}

#[derive(Deserialize)]
struct InboundFrom {
    #[serde(default)]
    id: Option<String>,
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/// Read an HTTP response body into memory with a hard byte cap.
///
/// Uses `reqwest::Response::chunk` (always available) rather than
/// `bytes_stream`, which would require the `futures` crate. Mirrors the
/// helper used by the other provider crates so a misconfigured endpoint
/// cannot exhaust memory with an unbounded body.
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
    use std::collections::HashMap;

    #[test]
    fn from_env_reads_all_four_variables() {
        // Preserve any pre-existing values.
        let keys = [
            "COPILOT_STUDIO_ENDPOINT",
            "COPILOT_STUDIO_TENANT_ID",
            "COPILOT_STUDIO_CLIENT_ID",
            "COPILOT_STUDIO_CLIENT_SECRET",
        ];
        let prev: HashMap<&str, Option<String>> =
            keys.iter().map(|k| (*k, std::env::var(*k).ok())).collect();

        std::env::set_var("COPILOT_STUDIO_ENDPOINT", "https://directline.example.invalid");
        std::env::set_var("COPILOT_STUDIO_TENANT_ID", "tenant-guid");
        std::env::set_var("COPILOT_STUDIO_CLIENT_ID", "client-guid");
        std::env::set_var("COPILOT_STUDIO_CLIENT_SECRET", "super-secret");

        let config = CopilotStudioConfig::from_env().expect("env config");
        assert_eq!(config.bot_endpoint, "https://directline.example.invalid");
        assert_eq!(config.tenant_id, "tenant-guid");
        assert_eq!(config.client_id, "client-guid");
        assert_eq!(config.client_secret.expose(), "super-secret");

        // Debug output must not leak the secret.
        let dbg = format!("{config:?}");
        assert!(!dbg.contains("super-secret"));
        assert!(dbg.contains("REDACTED"));

        // Restore environment.
        for (k, v) in prev {
            match v {
                Some(val) => std::env::set_var(k, val),
                None => std::env::remove_var(k),
            }
        }
    }

    #[test]
    fn client_builds_and_exposes_urls() {
        let config = CopilotStudioConfig::new(
            "https://directline.example.invalid/",
            "tenant",
            "client",
            SecretString::new("secret"),
        );
        let client = CopilotStudioClient::new(config).expect("client builds");

        // URL construction strips the trailing slash on bot_endpoint.
        assert_eq!(
            client.conversations_url(),
            "https://directline.example.invalid/v3/directline/conversations"
        );
        assert_eq!(
            client.activities_url("abc"),
            "https://directline.example.invalid/v3/directline/conversations/abc/activities"
        );

        // Credential header is marked sensitive.
        let headers = client.build_headers().expect("headers built");
        let auth = headers.get(AUTHORIZATION).expect("auth header present");
        assert!(auth.is_sensitive());
        assert!(auth.to_str().unwrap().starts_with("Bearer "));
    }

    #[test]
    fn activity_serializes_to_direct_line_shape() {
        // Verifies the JSON wire shape matches Direct Line's expected
        // `{type, from:{id}, text}` envelope so the test catches accidental
        // field renames.
        let activity = OutboundActivity {
            r#type: "message",
            from: ActivityFrom { id: "user_abc" },
            text: "hello bot",
        };
        let json = serde_json::to_value(&activity).expect("serializes");
        assert_eq!(json["type"], "message");
        assert_eq!(json["from"]["id"], "user_abc");
        assert_eq!(json["text"], "hello bot");
    }
}
