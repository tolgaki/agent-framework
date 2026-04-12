// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework Cosmos
//!
//! Azure Cosmos DB persistence for the Microsoft Agent Framework.
//!
//! Provides [`CosmosHistoryProvider`], a [`HistoryProvider`](agent_framework_core::session::HistoryProvider)
//! backed by the Cosmos DB REST API, and [`CosmosCheckpointStorage`] for
//! persisting workflow checkpoint state.
//!
//! # Example
//!
//! ```rust,no_run
//! use agent_framework_core::session::AgentSession;
//! use agent_framework_cosmos::{CosmosConfig, CosmosHistoryProvider};
//!
//! # async fn example() -> agent_framework_core::error::AgentResult<()> {
//! let config = CosmosConfig::from_env()?;
//! let provider = CosmosHistoryProvider::new(config)?;
//! let session = AgentSession::with_history_provider(Box::new(provider));
//! # Ok(())
//! # }
//! ```

use std::time::Duration;

use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use hmac::{Hmac, Mac};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use tracing::debug;

use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::http_limits::DEFAULT_MAX_RESPONSE_BYTES;
use agent_framework_core::redact::scrub_error_body;
use agent_framework_core::secret::SecretString;
use agent_framework_core::session::HistoryProvider;
use agent_framework_core::types::Message;

type HmacSha256 = Hmac<Sha256>;

const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for connecting to Azure Cosmos DB.
#[derive(Clone, Debug)]
pub struct CosmosConfig {
    /// The Cosmos DB account endpoint, e.g. `https://myaccount.documents.azure.com`.
    pub endpoint: String,

    /// The Cosmos DB master key (redacted in `Debug`).
    pub key: SecretString,

    /// Database name.
    pub database: String,

    /// Container name.
    pub container: String,

    /// Overall request timeout. Defaults to 30 seconds.
    pub request_timeout: Duration,

    /// TCP connect timeout. Defaults to 10 seconds.
    pub connect_timeout: Duration,
}

impl CosmosConfig {
    /// Create a config from environment variables.
    ///
    /// Reads `COSMOS_ENDPOINT`, `COSMOS_KEY` (required),
    /// `COSMOS_DATABASE`, and `COSMOS_CONTAINER`.
    pub fn from_env() -> AgentResult<Self> {
        let endpoint = std::env::var("COSMOS_ENDPOINT").map_err(|_| {
            AgentError::InvalidRequest("COSMOS_ENDPOINT environment variable is not set".into())
        })?;
        let key = std::env::var("COSMOS_KEY").map_err(|_| {
            AgentError::InvalidRequest("COSMOS_KEY environment variable is not set".into())
        })?;
        let database = std::env::var("COSMOS_DATABASE").map_err(|_| {
            AgentError::InvalidRequest("COSMOS_DATABASE environment variable is not set".into())
        })?;
        let container = std::env::var("COSMOS_CONTAINER").map_err(|_| {
            AgentError::InvalidRequest("COSMOS_CONTAINER environment variable is not set".into())
        })?;

        Ok(Self {
            endpoint,
            key: SecretString::new(key),
            database,
            container,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        })
    }
}

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

/// Generate a Cosmos DB master-key authorization token.
///
/// Format: `type=master&ver=1.0&sig={base64(HMAC-SHA256(key, payload))}`
///
/// The payload is: `"{verb}\n{resource_type}\n{resource_link}\n{date}\n\n"`
/// all lowercased except the date.
fn generate_auth_token(
    key: &SecretString,
    verb: &str,
    resource_type: &str,
    resource_link: &str,
    date: &str,
) -> AgentResult<String> {
    let key_bytes = BASE64.decode(key.expose().as_bytes()).map_err(|e| {
        AgentError::InvalidRequest(format!("Invalid Cosmos master key (base64): {e}"))
    })?;

    let payload = format!(
        "{}\n{}\n{}\n{}\n\n",
        verb.to_lowercase(),
        resource_type.to_lowercase(),
        resource_link.to_lowercase(),
        date.to_lowercase(),
    );

    let mut mac = HmacSha256::new_from_slice(&key_bytes).map_err(|e| {
        AgentError::InvalidRequest(format!("Invalid HMAC key length: {e}"))
    })?;
    mac.update(payload.as_bytes());
    let signature = BASE64.encode(mac.finalize().into_bytes());

    let token = format!("type=master&ver=1.0&sig={signature}");
    Ok(urlencoding::encode(&token).into_owned())
}

/// URL-encode a string for use in Cosmos auth headers.
/// Minimal encoder — only the chars that Cosmos cares about.
mod urlencoding {
    pub fn encode(s: &str) -> std::borrow::Cow<'_, str> {
        let needs_encoding = s.bytes().any(|b| {
            !b.is_ascii_alphanumeric()
                && b != b'-'
                && b != b'_'
                && b != b'.'
                && b != b'~'
        });
        if !needs_encoding {
            return std::borrow::Cow::Borrowed(s);
        }
        let mut out = String::with_capacity(s.len() * 3);
        for b in s.bytes() {
            if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.' || b == b'~' {
                out.push(b as char);
            } else {
                out.push('%');
                out.push_str(&format!("{b:02X}"));
            }
        }
        std::borrow::Cow::Owned(out)
    }
}

/// Get the current UTC date in RFC 7231 format for the `x-ms-date` header.
fn rfc7231_date() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    // Manual formatting to avoid pulling in chrono.
    let secs = now.as_secs();
    // Use a simple approach: format via the libc-free method.
    // Cosmos accepts any HTTP-date. We produce the preferred format.
    format_http_date(secs)
}

fn format_http_date(epoch_secs: u64) -> String {
    // Days/months tables for HTTP-date (RFC 7231).
    const DAYS: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];

    // Seconds from Unix epoch.
    let total_days = (epoch_secs / 86400) as i64;
    let day_seconds = (epoch_secs % 86400) as u32;
    let hour = day_seconds / 3600;
    let minute = (day_seconds % 3600) / 60;
    let second = day_seconds % 60;

    // Day of week: 1970-01-01 was Thursday (index 0).
    let dow = ((total_days % 7 + 7) % 7) as usize;

    // Convert total_days to y/m/d via a civil-calendar algorithm.
    let (year, month, day) = civil_from_days(total_days);

    format!(
        "{}, {:02} {} {:04} {:02}:{:02}:{:02} GMT",
        DAYS[dow],
        day,
        MONTHS[month as usize - 1],
        year,
        hour,
        minute,
        second,
    )
}

/// Convert days since epoch to (year, month, day). Algorithm from Howard
/// Hinnant's `civil_from_days`.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m as u32, d as u32)
}

// ---------------------------------------------------------------------------
// Cosmos document shape
// ---------------------------------------------------------------------------

/// Wrapper for a history document stored in Cosmos DB.
#[derive(Serialize, Deserialize)]
struct HistoryDocument {
    /// Cosmos DB requires an `id` field as the document primary key.
    id: String,
    /// The serialized conversation messages.
    messages: Vec<Message>,
}

/// Wrapper for a checkpoint document stored in Cosmos DB.
#[derive(Serialize, Deserialize)]
struct CheckpointDocument {
    id: String,
    data: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Helper: read a bounded response body
// ---------------------------------------------------------------------------

async fn read_bounded_body(resp: reqwest::Response) -> AgentResult<String> {
    let status = resp.status();
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| AgentError::HttpError(format!("Failed to read response body: {e}")))?;

    if bytes.len() > DEFAULT_MAX_RESPONSE_BYTES {
        return Err(AgentError::HttpError(format!(
            "Response body exceeds {DEFAULT_MAX_RESPONSE_BYTES} bytes"
        )));
    }

    let body = String::from_utf8_lossy(&bytes).into_owned();

    if !status.is_success() {
        return Err(AgentError::provider(
            format!("Cosmos DB returned {status}: {}", scrub_error_body(&body)),
            Some(status.as_u16()),
        ));
    }
    Ok(body)
}

// ---------------------------------------------------------------------------
// CosmosHistoryProvider
// ---------------------------------------------------------------------------

/// A [`HistoryProvider`] backed by Azure Cosmos DB.
///
/// Stores conversation history as JSON documents using the Cosmos DB REST API.
/// Each session gets a single document keyed by `session_id`.
#[derive(Debug)]
pub struct CosmosHistoryProvider {
    config: CosmosConfig,
    client: reqwest::Client,
}

impl CosmosHistoryProvider {
    /// Create a new provider from the given config.
    pub fn new(config: CosmosConfig) -> AgentResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .connect_timeout(config.connect_timeout)
            .build()
            .map_err(|e| AgentError::HttpError(format!("Failed to build HTTP client: {e}")))?;
        Ok(Self { config, client })
    }

    /// Build default headers for a Cosmos REST request.
    fn build_headers(
        &self,
        verb: &str,
        resource_type: &str,
        resource_link: &str,
    ) -> AgentResult<(HeaderMap, String)> {
        let date = rfc7231_date();
        let auth =
            generate_auth_token(&self.config.key, verb, resource_type, resource_link, &date)?;

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-ms-date",
            HeaderValue::from_str(&date)
                .map_err(|e| AgentError::HttpError(format!("Invalid date header: {e}")))?,
        );
        headers.insert(
            "x-ms-version",
            HeaderValue::from_static("2018-12-31"),
        );
        headers.insert(
            "Authorization",
            HeaderValue::from_str(&auth)
                .map_err(|e| AgentError::HttpError(format!("Invalid auth header: {e}")))?,
        );
        Ok((headers, date))
    }

    /// Base URL for the docs collection.
    fn docs_url(&self) -> String {
        let ep = self.config.endpoint.trim_end_matches('/');
        format!(
            "{}/dbs/{}/colls/{}/docs",
            ep, self.config.database, self.config.container
        )
    }

    /// Resource link for a specific document.
    fn doc_resource_link(&self, doc_id: &str) -> String {
        format!(
            "dbs/{}/colls/{}/docs/{}",
            self.config.database, self.config.container, doc_id
        )
    }

    /// Resource link for the docs collection.
    fn colls_resource_link(&self) -> String {
        format!(
            "dbs/{}/colls/{}",
            self.config.database, self.config.container
        )
    }
}

#[async_trait]
impl HistoryProvider for CosmosHistoryProvider {
    async fn get_history(&self, session_id: &str) -> AgentResult<Vec<Message>> {
        let resource_link = self.doc_resource_link(session_id);
        let (headers, _) = self.build_headers("get", "docs", &resource_link)?;
        let url = format!("{}/{}", self.docs_url(), session_id);

        debug!(session_id, "Cosmos: fetching history");

        let resp = self
            .client
            .get(&url)
            .headers(headers)
            .send()
            .await
            .map_err(|e| AgentError::HttpError(format!("Cosmos GET failed: {e}")))?;

        // 404 means no history yet — return empty.
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(Vec::new());
        }

        let body = read_bounded_body(resp).await?;
        let doc: HistoryDocument = serde_json::from_str(&body).map_err(|e| {
            AgentError::InvalidResponse(format!("Failed to parse history document: {e}"))
        })?;
        Ok(doc.messages)
    }

    async fn save_history(&self, session_id: &str, messages: &[Message]) -> AgentResult<()> {
        // Read existing history, merge, and upsert.
        let mut existing = self.get_history(session_id).await.unwrap_or_default();
        existing.extend(messages.iter().cloned());

        let doc = HistoryDocument {
            id: session_id.to_string(),
            messages: existing,
        };

        let resource_link = self.colls_resource_link();
        let (mut headers, _) = self.build_headers("post", "docs", &resource_link)?;
        // Enable upsert.
        headers.insert(
            "x-ms-documentdb-is-upsert",
            HeaderValue::from_static("true"),
        );

        let url = self.docs_url();
        let body = serde_json::to_string(&doc)?;

        debug!(session_id, "Cosmos: saving history ({} messages)", doc.messages.len());

        let resp = self
            .client
            .post(&url)
            .headers(headers)
            .body(body)
            .send()
            .await
            .map_err(|e| AgentError::HttpError(format!("Cosmos POST (upsert) failed: {e}")))?;

        let _ = read_bounded_body(resp).await?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CosmosCheckpointStorage
// ---------------------------------------------------------------------------

/// A checkpoint storage backend backed by Azure Cosmos DB.
///
/// Stores arbitrary JSON checkpoint data as Cosmos documents keyed by a
/// caller-supplied checkpoint ID. This type mirrors the `CheckpointStorage`
/// pattern from the workflows crate but is defined locally to avoid a hard
/// dependency on `agent-framework-workflows`.
#[derive(Debug)]
pub struct CosmosCheckpointStorage {
    config: CosmosConfig,
    client: reqwest::Client,
}

impl CosmosCheckpointStorage {
    /// Create a new checkpoint storage from the given config.
    pub fn new(config: CosmosConfig) -> AgentResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .connect_timeout(config.connect_timeout)
            .build()
            .map_err(|e| AgentError::HttpError(format!("Failed to build HTTP client: {e}")))?;
        Ok(Self { config, client })
    }

    /// Base URL for the docs collection.
    fn docs_url(&self) -> String {
        let ep = self.config.endpoint.trim_end_matches('/');
        format!(
            "{}/dbs/{}/colls/{}/docs",
            ep, self.config.database, self.config.container
        )
    }

    fn doc_resource_link(&self, doc_id: &str) -> String {
        format!(
            "dbs/{}/colls/{}/docs/{}",
            self.config.database, self.config.container, doc_id
        )
    }

    fn colls_resource_link(&self) -> String {
        format!(
            "dbs/{}/colls/{}",
            self.config.database, self.config.container
        )
    }

    fn build_headers(
        &self,
        verb: &str,
        resource_type: &str,
        resource_link: &str,
    ) -> AgentResult<HeaderMap> {
        let date = rfc7231_date();
        let auth =
            generate_auth_token(&self.config.key, verb, resource_type, resource_link, &date)?;

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-ms-date",
            HeaderValue::from_str(&date)
                .map_err(|e| AgentError::HttpError(format!("Invalid date header: {e}")))?,
        );
        headers.insert(
            "x-ms-version",
            HeaderValue::from_static("2018-12-31"),
        );
        headers.insert(
            "Authorization",
            HeaderValue::from_str(&auth)
                .map_err(|e| AgentError::HttpError(format!("Invalid auth header: {e}")))?,
        );
        Ok(headers)
    }

    /// Save a checkpoint document.
    ///
    /// Upserts the document (creates or replaces).
    pub async fn save(&self, checkpoint_id: &str, data: serde_json::Value) -> AgentResult<()> {
        let doc = CheckpointDocument {
            id: checkpoint_id.to_string(),
            data,
        };

        let resource_link = self.colls_resource_link();
        let mut headers = self.build_headers("post", "docs", &resource_link)?;
        headers.insert(
            "x-ms-documentdb-is-upsert",
            HeaderValue::from_static("true"),
        );

        let url = self.docs_url();
        let body = serde_json::to_string(&doc)?;

        debug!(checkpoint_id, "Cosmos: saving checkpoint");

        let resp = self
            .client
            .post(&url)
            .headers(headers)
            .body(body)
            .send()
            .await
            .map_err(|e| AgentError::HttpError(format!("Cosmos POST (upsert) failed: {e}")))?;

        let _ = read_bounded_body(resp).await?;
        Ok(())
    }

    /// Load a checkpoint document.
    ///
    /// Returns `None` if the checkpoint does not exist.
    pub async fn load(&self, checkpoint_id: &str) -> AgentResult<Option<serde_json::Value>> {
        let resource_link = self.doc_resource_link(checkpoint_id);
        let headers = self.build_headers("get", "docs", &resource_link)?;
        let url = format!("{}/{}", self.docs_url(), checkpoint_id);

        debug!(checkpoint_id, "Cosmos: loading checkpoint");

        let resp = self
            .client
            .get(&url)
            .headers(headers)
            .send()
            .await
            .map_err(|e| AgentError::HttpError(format!("Cosmos GET failed: {e}")))?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }

        let body = read_bounded_body(resp).await?;
        let doc: CheckpointDocument = serde_json::from_str(&body).map_err(|e| {
            AgentError::InvalidResponse(format!("Failed to parse checkpoint document: {e}"))
        })?;
        Ok(Some(doc.data))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests env-var reading. Combines missing + success cases in a single
    /// test to avoid data races when parallel tests mutate the environment.
    #[test]
    fn config_from_env() {
        // --- missing vars ---
        std::env::remove_var("COSMOS_ENDPOINT");
        std::env::remove_var("COSMOS_KEY");
        std::env::remove_var("COSMOS_DATABASE");
        std::env::remove_var("COSMOS_CONTAINER");

        let err = CosmosConfig::from_env().unwrap_err();
        assert!(
            err.to_string().contains("COSMOS_ENDPOINT"),
            "Expected COSMOS_ENDPOINT error, got: {err}"
        );

        // --- all vars set ---
        std::env::set_var("COSMOS_ENDPOINT", "https://test.documents.azure.com");
        std::env::set_var("COSMOS_KEY", "dGVzdGtleQ==");
        std::env::set_var("COSMOS_DATABASE", "testdb");
        std::env::set_var("COSMOS_CONTAINER", "testcontainer");

        let config = CosmosConfig::from_env().unwrap();
        assert_eq!(config.endpoint, "https://test.documents.azure.com");
        assert_eq!(config.key.expose(), "dGVzdGtleQ==");
        assert_eq!(config.database, "testdb");
        assert_eq!(config.container, "testcontainer");

        // Clean up.
        std::env::remove_var("COSMOS_ENDPOINT");
        std::env::remove_var("COSMOS_KEY");
        std::env::remove_var("COSMOS_DATABASE");
        std::env::remove_var("COSMOS_CONTAINER");
    }

    #[test]
    fn config_debug_redacts_key() {
        let config = CosmosConfig {
            endpoint: "https://test.documents.azure.com".into(),
            key: SecretString::new("super-secret"),
            database: "db".into(),
            container: "coll".into(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        };
        let debug = format!("{config:?}");
        assert!(!debug.contains("super-secret"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn generate_auth_token_produces_expected_format() {
        // Use a known base64-encoded key.
        let key = SecretString::new("dGVzdGtleQ=="); // base64("testkey")
        let token = generate_auth_token(&key, "get", "docs", "dbs/db/colls/coll/docs/id1", "Mon, 01 Jan 2024 00:00:00 GMT")
            .unwrap();
        // Token should be URL-encoded and start with type%3Dmaster.
        assert!(token.starts_with("type%3Dmaster%26ver%3D1.0%26sig%3D"), "Got: {token}");
    }

    #[test]
    fn rfc7231_date_has_correct_format() {
        let date = rfc7231_date();
        // Should look like: "Thu, 01 Jan 1970 00:00:00 GMT" (but current time).
        assert!(date.ends_with(" GMT"), "Date should end with GMT: {date}");
        assert!(date.len() > 20, "Date too short: {date}");
    }

    #[test]
    fn format_http_date_known_epoch() {
        // 2024-01-01 00:00:00 UTC = epoch 1704067200
        let date = format_http_date(1_704_067_200);
        assert_eq!(date, "Mon, 01 Jan 2024 00:00:00 GMT");
    }

    #[test]
    fn docs_url_formation() {
        let config = CosmosConfig {
            endpoint: "https://myaccount.documents.azure.com".into(),
            key: SecretString::new("dGVzdA=="),
            database: "mydb".into(),
            container: "mycoll".into(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        };
        let provider = CosmosHistoryProvider::new(config).unwrap();
        assert_eq!(
            provider.docs_url(),
            "https://myaccount.documents.azure.com/dbs/mydb/colls/mycoll/docs"
        );
        assert_eq!(
            provider.doc_resource_link("session-1"),
            "dbs/mydb/colls/mycoll/docs/session-1"
        );
    }

    #[test]
    fn history_document_roundtrip() {
        let doc = HistoryDocument {
            id: "s1".into(),
            messages: vec![Message::user("hello"), Message::assistant("hi")],
        };
        let json = serde_json::to_string(&doc).unwrap();
        let parsed: HistoryDocument = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "s1");
        assert_eq!(parsed.messages.len(), 2);
    }

    #[test]
    fn checkpoint_document_roundtrip() {
        let doc = CheckpointDocument {
            id: "cp-1".into(),
            data: serde_json::json!({"step": 3, "state": "running"}),
        };
        let json = serde_json::to_string(&doc).unwrap();
        let parsed: CheckpointDocument = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "cp-1");
        assert_eq!(parsed.data["step"], 3);
    }
}
