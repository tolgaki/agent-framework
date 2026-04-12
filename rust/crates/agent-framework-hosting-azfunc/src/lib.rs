// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework Azure Functions Hosting
//!
//! Azure Functions Custom Handler hosting for the Microsoft Agent Framework.
//!
//! Mirrors .NET's `Microsoft.Agents.AI.Hosting.AzureFunctions`.
//!
//! # Azure Functions custom handlers
//!
//! Azure Functions Custom Handlers work by launching a user-provided HTTP
//! server on a port specified by the `FUNCTIONS_CUSTOMHANDLER_PORT`
//! environment variable. The Functions host forwards HTTP trigger invocations
//! to this server and translates the JSON payload into a custom envelope.
//!
//! This crate wraps [`agent_framework_hosting::build_router`] with additional
//! Azure Functions specific routes:
//!
//! - `POST /api/agent` -- the Azure Functions HTTP trigger entry point
//! - `GET /api/health` -- liveness/health probe
//! - `/api/v1/...` -- the full set of agent endpoints re-exposed under `/api`
//!
//! # Example
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use agent_framework_hosting_azfunc::{AzureFunctionsConfig, AzureFunctionsHost};
//! # async fn example(agent: Arc<dyn agent_framework_core::agent::Agent>) {
//! let config = AzureFunctionsConfig::from_env();
//! let host = AzureFunctionsHost::from_arc(agent);
//! host.serve(config).await.unwrap();
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use serde::{Deserialize, Serialize};
use tracing::debug;

use agent_framework_core::agent::Agent;
use agent_framework_core::error::AgentError;
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::Message;
use agent_framework_hosting::build_router;

/// Environment variable used by the Azure Functions host to communicate the
/// port that the custom handler must bind to.
pub const FUNCTIONS_CUSTOMHANDLER_PORT_ENV: &str = "FUNCTIONS_CUSTOMHANDLER_PORT";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Azure Functions custom handler host.
#[derive(Debug, Clone)]
pub struct AzureFunctionsConfig {
    /// The port to bind to.
    ///
    /// Defaults to `FUNCTIONS_CUSTOMHANDLER_PORT` when constructed via
    /// [`from_env`](Self::from_env), or `8080` if the env var is missing.
    pub port: u16,
}

impl AzureFunctionsConfig {
    /// Read the Azure Functions custom handler port from the environment,
    /// falling back to `8080` if `FUNCTIONS_CUSTOMHANDLER_PORT` is unset or
    /// cannot be parsed as a `u16`.
    pub fn from_env() -> Self {
        let port = std::env::var(FUNCTIONS_CUSTOMHANDLER_PORT_ENV)
            .ok()
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(8080);
        Self { port }
    }
}

impl Default for AzureFunctionsConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

// ---------------------------------------------------------------------------
// Custom handler envelope types
// ---------------------------------------------------------------------------

/// The Azure Functions Custom Handler request envelope.
///
/// The Functions host POSTs a JSON body of the form:
///
/// ```json
/// {
///   "Data": {
///     "req": {
///       "Body": "<json-string>",
///       "Headers": { "header-name": "value" }
///     }
///   },
///   "Metadata": { ... }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureFunctionsRequest {
    /// Bindings data. The HTTP trigger lives under `req`.
    #[serde(rename = "Data")]
    pub data: AzureFunctionsRequestData,

    /// Invocation metadata supplied by the Functions runtime.
    #[serde(rename = "Metadata", default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// The `Data` section of an [`AzureFunctionsRequest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureFunctionsRequestData {
    /// The HTTP request trigger, keyed as `req` (matching the binding name in
    /// `function.json`).
    pub req: AzureFunctionsHttpRequest,
}

/// The HTTP trigger payload inside an Azure Functions Custom Handler request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureFunctionsHttpRequest {
    /// The raw body as a JSON-encoded string.
    #[serde(rename = "Body", default)]
    pub body: String,

    /// The request headers.
    #[serde(rename = "Headers", default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,

    /// The HTTP method, when provided by the host.
    #[serde(rename = "Method", default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,

    /// The request URL, when provided by the host.
    #[serde(rename = "Url", default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

/// The Azure Functions Custom Handler response envelope.
///
/// The Functions host expects:
///
/// ```json
/// {
///   "Outputs": {
///     "res": {
///       "Body": "<json-string>",
///       "StatusCode": "200",
///       "Headers": { "Content-Type": "application/json" }
///     }
///   },
///   "Logs": [],
///   "ReturnValue": null
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureFunctionsResponse {
    /// Binding outputs. The HTTP response lives under `res`.
    #[serde(rename = "Outputs")]
    pub outputs: AzureFunctionsResponseOutputs,

    /// Log lines to surface in the Functions host logs.
    #[serde(rename = "Logs", default, skip_serializing_if = "Vec::is_empty")]
    pub logs: Vec<String>,

    /// The handler return value (unused).
    #[serde(rename = "ReturnValue", default, skip_serializing_if = "Option::is_none")]
    pub return_value: Option<serde_json::Value>,
}

/// The `Outputs` section of an [`AzureFunctionsResponse`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureFunctionsResponseOutputs {
    /// The HTTP response, keyed as `res` (matching the binding name in
    /// `function.json`).
    pub res: AzureFunctionsHttpResponse,
}

/// The HTTP response payload inside an Azure Functions Custom Handler response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AzureFunctionsHttpResponse {
    /// The response body, typically a JSON-encoded string.
    #[serde(rename = "Body")]
    pub body: String,

    /// The HTTP status code, as a string (per the custom handler contract).
    #[serde(rename = "StatusCode")]
    pub status_code: String,

    /// The response headers.
    #[serde(rename = "Headers", default, skip_serializing_if = "HashMap::is_empty")]
    pub headers: HashMap<String, String>,
}

impl AzureFunctionsResponse {
    /// Create a JSON success response.
    pub fn ok_json(body: impl Into<String>) -> Self {
        Self::json(200, body)
    }

    /// Create a JSON response with the given status code.
    pub fn json(status: u16, body: impl Into<String>) -> Self {
        let mut headers = HashMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        Self {
            outputs: AzureFunctionsResponseOutputs {
                res: AzureFunctionsHttpResponse {
                    body: body.into(),
                    status_code: status.to_string(),
                    headers,
                },
            },
            logs: Vec::new(),
            return_value: None,
        }
    }

    /// Create a plain-text error response with the given status code.
    pub fn error(status: u16, message: impl Into<String>) -> Self {
        let body = serde_json::json!({
            "error": {
                "message": message.into(),
                "type": "agent_error",
            }
        })
        .to_string();
        Self::json(status, body)
    }
}

// ---------------------------------------------------------------------------
// Input body that clients POST via the Functions trigger.
// ---------------------------------------------------------------------------

/// The JSON body clients are expected to pass through the Azure Functions
/// HTTP trigger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInvokeBody {
    /// Messages to send to the agent.
    pub messages: Vec<Message>,

    /// Optional session identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Functional entry point
// ---------------------------------------------------------------------------

/// Function-style entry point: given an agent and a parsed custom-handler
/// request, runs the agent and produces a custom-handler response.
///
/// This is useful when embedding the handler into an external HTTP framework
/// or when writing unit tests that do not want to spin up a full server.
pub async fn host_function_handler(
    agent: Arc<dyn Agent>,
    req: AzureFunctionsRequest,
) -> AzureFunctionsResponse {
    debug!(
        method = req.data.req.method.as_deref().unwrap_or("?"),
        url = req.data.req.url.as_deref().unwrap_or("?"),
        "Azure Functions handler invoked"
    );

    let body: AgentInvokeBody = match serde_json::from_str(&req.data.req.body) {
        Ok(b) => b,
        Err(e) => {
            debug!(error = %e, "Failed to parse agent invoke body");
            return AzureFunctionsResponse::error(400, format!("invalid request body: {e}"));
        }
    };

    let mut session = AgentSession::new();
    if let Some(sid) = body.session_id {
        session.session_id = sid;
    }

    match agent.run(body.messages, &mut session, None).await {
        Ok(response) => match serde_json::to_string(&response) {
            Ok(json) => AzureFunctionsResponse::ok_json(json),
            Err(e) => AzureFunctionsResponse::error(500, format!("serialization error: {e}")),
        },
        Err(err) => {
            let status = match &err {
                AgentError::InvalidRequest(_) => 400,
                AgentError::ProviderError { status_code, .. } => status_code.unwrap_or(502),
                _ => 500,
            };
            AzureFunctionsResponse::error(status, err.to_string())
        }
    }
}

// ---------------------------------------------------------------------------
// Host (router + server)
// ---------------------------------------------------------------------------

/// Shared state for the Azure Functions host.
#[derive(Clone)]
struct AzFuncState {
    agent: Arc<dyn Agent>,
}

/// Azure Functions Custom Handler host that exposes an [`Agent`] over HTTP.
///
/// Wraps [`agent_framework_hosting::build_router`] with additional
/// Azure-Functions-specific routes (`/api/agent`, `/api/health`, and the
/// `/v1/...` agent routes re-exposed under `/api/v1/...`).
pub struct AzureFunctionsHost {
    agent: Arc<dyn Agent>,
}

impl AzureFunctionsHost {
    /// Create a host from any concrete [`Agent`] implementation.
    pub fn new(agent: impl Agent + 'static) -> Self {
        Self {
            agent: Arc::new(agent),
        }
    }

    /// Create a host from a pre-wrapped `Arc<dyn Agent>`.
    pub fn from_arc(agent: Arc<dyn Agent>) -> Self {
        Self { agent }
    }

    /// Build the axum [`Router`] for this host.
    ///
    /// Routes:
    /// - `POST /api/agent` -- Azure Functions trigger entry point
    /// - `GET  /api/health` -- health probe
    /// - `POST /api/v1/agent/run`, `POST /api/v1/agent/run/stream`,
    ///   `POST /api/v1/chat/completions`, `GET /api/v1/agent/info` --
    ///   the standard agent routes, mounted under `/api`
    /// - The same agent routes under their native `/v1/...` paths
    pub fn router(&self) -> Router {
        let state = AzFuncState {
            agent: Arc::clone(&self.agent),
        };

        // The native `/v1/...` router from the base hosting crate.
        let v1 = build_router(Arc::clone(&self.agent));

        // Mount the same v1 routes under `/api` so Azure Functions HTTP
        // triggers (which traditionally use `/api/...` paths) can reach them.
        let api_v1 = build_router(Arc::clone(&self.agent));

        Router::new()
            .route("/api/agent", post(handle_agent_trigger))
            .route("/api/health", get(handle_health))
            .with_state(state)
            .nest("/api", api_v1)
            .merge(v1)
    }

    /// Bind and serve the host on the configured port.
    pub async fn serve(self, config: AzureFunctionsConfig) -> Result<(), AgentError> {
        let router = self.router();
        let addr = format!("0.0.0.0:{}", config.port);

        debug!(addr = %addr, "Starting Azure Functions custom handler host");

        let listener = tokio::net::TcpListener::bind(&addr)
            .await
            .map_err(|e| AgentError::HttpError(format!("failed to bind to {addr}: {e}")))?;

        axum::serve(listener, router)
            .await
            .map_err(|e| AgentError::HttpError(format!("server error: {e}")))?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

/// `GET /api/health` -- always returns 200 with a tiny JSON body.
async fn handle_health() -> Response {
    Json(serde_json::json!({ "status": "ok" })).into_response()
}

/// `POST /api/agent` -- Azure Functions custom handler entry point.
async fn handle_agent_trigger(
    State(state): State<AzFuncState>,
    Json(req): Json<AzureFunctionsRequest>,
) -> Json<AzureFunctionsResponse> {
    let response = host_function_handler(state.agent, req).await;
    Json(response)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_from_env_uses_env_var() {
        // Snapshot and restore the environment variable to avoid leaking
        // state into other tests running in the same process.
        let previous = std::env::var(FUNCTIONS_CUSTOMHANDLER_PORT_ENV).ok();

        // SAFETY: tests in this crate do not run concurrently with other
        // code that reads this variable.
        unsafe {
            std::env::set_var(FUNCTIONS_CUSTOMHANDLER_PORT_ENV, "12345");
        }
        let config = AzureFunctionsConfig::from_env();
        assert_eq!(config.port, 12345);

        unsafe {
            std::env::remove_var(FUNCTIONS_CUSTOMHANDLER_PORT_ENV);
        }
        let default_config = AzureFunctionsConfig::from_env();
        assert_eq!(default_config.port, 8080);

        // Restore.
        if let Some(v) = previous {
            unsafe {
                std::env::set_var(FUNCTIONS_CUSTOMHANDLER_PORT_ENV, v);
            }
        }
    }

    #[test]
    fn request_deserializes_custom_handler_envelope() {
        let raw = r#"{
            "Data": {
                "req": {
                    "Body": "{\"messages\":[]}",
                    "Headers": { "content-type": "application/json" },
                    "Method": "POST",
                    "Url": "http://localhost/api/agent"
                }
            },
            "Metadata": {}
        }"#;

        let req: AzureFunctionsRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.data.req.body, "{\"messages\":[]}");
        assert_eq!(req.data.req.method.as_deref(), Some("POST"));
        assert_eq!(
            req.data.req.headers.get("content-type").map(String::as_str),
            Some("application/json")
        );
    }

    #[test]
    fn response_serializes_to_custom_handler_envelope() {
        let response = AzureFunctionsResponse::ok_json("{\"text\":\"hi\"}".to_string());
        let json = serde_json::to_value(&response).unwrap();

        assert_eq!(json["Outputs"]["res"]["StatusCode"], "200");
        assert_eq!(json["Outputs"]["res"]["Body"], "{\"text\":\"hi\"}");
        assert_eq!(
            json["Outputs"]["res"]["Headers"]["Content-Type"],
            "application/json"
        );
    }

    #[test]
    fn error_response_contains_status_and_message() {
        let response = AzureFunctionsResponse::error(400, "bad request");
        assert_eq!(response.outputs.res.status_code, "400");
        assert!(response.outputs.res.body.contains("bad request"));
    }

    #[test]
    fn default_config_falls_back_without_env() {
        let previous = std::env::var(FUNCTIONS_CUSTOMHANDLER_PORT_ENV).ok();
        unsafe {
            std::env::remove_var(FUNCTIONS_CUSTOMHANDLER_PORT_ENV);
        }
        let config = AzureFunctionsConfig::default();
        assert_eq!(config.port, 8080);
        if let Some(v) = previous {
            unsafe {
                std::env::set_var(FUNCTIONS_CUSTOMHANDLER_PORT_ENV, v);
            }
        }
    }
}
