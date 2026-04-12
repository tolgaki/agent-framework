// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework Hosting
//!
//! HTTP hosting for the Microsoft Agent Framework.
//!
//! This crate provides [`AgentServer`] and [`serve`], which expose any
//! [`Agent`](agent_framework_core::agent::Agent) as an HTTP API with:
//!
//! - `POST /v1/agent/run` -- native agent run endpoint
//! - `POST /v1/agent/run/stream` -- native agent streaming endpoint (SSE)
//! - `POST /v1/chat/completions` -- OpenAI-compatible completions endpoint
//! - `GET /v1/agent/info` -- agent metadata
//!
//! # Example
//!
//! ```rust,no_run
//! use agent_framework_hosting::{AgentServerConfig, serve};
//! // Assume `my_agent` implements the Agent trait.
//! # async fn example(my_agent: impl agent_framework_core::agent::Agent + 'static) {
//! let config = AgentServerConfig::default();
//! serve(my_agent, config).await.unwrap();
//! # }
//! ```

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;
use tracing::{debug, error};

use agent_framework_core::agent::Agent;
use agent_framework_core::error::AgentError;
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::{
    AgentResponse, AgentRunOptions, ChatOptions, FinishReason, Message, Usage,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the agent HTTP server.
#[derive(Debug, Clone)]
pub struct AgentServerConfig {
    /// The host address to bind to. Defaults to `"0.0.0.0"`.
    pub host: String,

    /// The port to listen on. Defaults to `8080`.
    pub port: u16,
}

impl Default for AgentServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
        }
    }
}

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

/// Request body for the native `/v1/agent/run` and `/v1/agent/run/stream` endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunRequest {
    /// The input messages for the agent.
    pub messages: Vec<Message>,

    /// Optional session ID. If not provided, a new session is created.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,

    /// Optional per-call overrides.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<AgentRunRequestOptions>,
}

/// Serializable options for agent run requests.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentRunRequestOptions {
    /// Per-call chat option overrides.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_options: Option<ChatOptions>,

    /// Additional system-level instructions for this call only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub additional_instructions: Option<String>,
}

impl From<AgentRunRequestOptions> for AgentRunOptions {
    fn from(opts: AgentRunRequestOptions) -> Self {
        Self {
            chat_options: opts.chat_options,
            additional_instructions: opts.additional_instructions,
        }
    }
}

/// Response body for the native `/v1/agent/run` endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunResponse {
    /// The final text output from the agent.
    pub text: String,

    /// All messages produced during the agent run.
    pub messages: Vec<Message>,

    /// The reason the model stopped generating.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Token usage accumulated across all model calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl From<AgentResponse> for AgentRunResponse {
    fn from(resp: AgentResponse) -> Self {
        Self {
            text: resp.text,
            messages: resp.messages,
            finish_reason: resp.finish_reason,
            usage: resp.usage,
        }
    }
}

/// SSE event emitted by the `/v1/agent/run/stream` endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvent {
    /// Incremental text, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// The finish reason, if this is the final event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Usage info, typically sent with the final event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Response body for the `/v1/agent/info` endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfoResponse {
    /// The agent's unique identifier.
    pub id: String,

    /// The agent's human-readable name, if set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// The agent's description, if set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

// ---------------------------------------------------------------------------
// OpenAI-compatible types
// ---------------------------------------------------------------------------

/// Request body for the OpenAI-compatible `/v1/chat/completions` endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// The messages to send to the agent.
    pub messages: Vec<ChatCompletionMessage>,

    /// Model identifier (accepted but not used to select the agent).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

/// A message in OpenAI chat completion format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionMessage {
    /// The role of the message author.
    pub role: String,

    /// The message content.
    pub content: String,
}

impl ChatCompletionMessage {
    /// Convert to the framework's [`Message`] type.
    fn into_message(self) -> Message {
        match self.role.as_str() {
            "system" => Message::system(self.content),
            "assistant" => Message::assistant(self.content),
            _ => Message::user(self.content),
        }
    }
}

/// Response body for the OpenAI-compatible `/v1/chat/completions` endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    /// A unique identifier for the completion.
    pub id: String,

    /// The object type (always `"chat.completion"`).
    pub object: String,

    /// The list of completion choices.
    pub choices: Vec<ChatCompletionChoice>,

    /// Token usage statistics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ChatCompletionUsage>,
}

/// A single choice in a chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    /// The index of this choice.
    pub index: u32,

    /// The generated message.
    pub message: ChatCompletionMessage,

    /// The reason the model stopped generating.
    pub finish_reason: String,
}

/// Usage statistics in OpenAI format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionUsage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,

    /// Number of tokens in the completion.
    pub completion_tokens: u32,

    /// Total tokens used.
    pub total_tokens: u32,
}

// ---------------------------------------------------------------------------
// Error response
// ---------------------------------------------------------------------------

/// JSON error response body.
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
}

fn error_response(status: axum::http::StatusCode, message: impl Into<String>) -> Response {
    let body = ErrorResponse {
        error: ErrorDetail {
            message: message.into(),
            error_type: "agent_error".to_string(),
        },
    };
    (status, Json(body)).into_response()
}

fn agent_error_to_response(err: AgentError) -> Response {
    let status = match &err {
        AgentError::InvalidRequest(_) => axum::http::StatusCode::BAD_REQUEST,
        AgentError::ProviderError { status_code, .. } => {
            axum::http::StatusCode::from_u16(status_code.unwrap_or(502))
                .unwrap_or(axum::http::StatusCode::BAD_GATEWAY)
        }
        _ => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
    };
    error!(error = %err, "Agent error");
    error_response(status, err.to_string())
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// Shared application state holding the agent instance.
#[derive(Clone)]
struct AppState {
    agent: Arc<dyn Agent>,
}

// ---------------------------------------------------------------------------
// Route handlers
// ---------------------------------------------------------------------------

/// GET /v1/agent/info
async fn handle_agent_info(State(state): State<AppState>) -> Json<AgentInfoResponse> {
    Json(AgentInfoResponse {
        id: state.agent.id().to_string(),
        name: state.agent.name().map(|s| s.to_string()),
        description: state.agent.description().map(|s| s.to_string()),
    })
}

/// POST /v1/agent/run
async fn handle_agent_run(
    State(state): State<AppState>,
    Json(request): Json<AgentRunRequest>,
) -> Response {
    debug!(messages = request.messages.len(), "Agent run request");

    let mut session = AgentSession::new();
    if let Some(sid) = &request.session_id {
        session.session_id = sid.clone();
    }

    let options = request.options.map(AgentRunOptions::from);
    let result = state
        .agent
        .run(request.messages, &mut session, options.as_ref())
        .await;

    match result {
        Ok(response) => {
            let run_response: AgentRunResponse = response.into();
            Json(run_response).into_response()
        }
        Err(err) => agent_error_to_response(err),
    }
}

/// POST /v1/agent/run/stream
async fn handle_agent_run_stream(
    State(state): State<AppState>,
    Json(request): Json<AgentRunRequest>,
) -> Response {
    debug!(messages = request.messages.len(), "Agent stream request");

    // We need the session to live as long as the stream. Since the Agent trait's
    // run_stream borrows session, we use a tokio task + channel to decouple
    // lifetimes.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);

    tokio::spawn(async move {
        let mut session = AgentSession::new();
        if let Some(sid) = &request.session_id {
            session.session_id = sid.clone();
        }

        let options = request.options.map(AgentRunOptions::from);
        let stream_result = state
            .agent
            .run_stream(request.messages, &mut session, options.as_ref());

        let mut stream = match stream_result {
            Ok(s) => s,
            Err(err) => {
                error!(error = %err, "Failed to create agent stream");
                return;
            }
        };

        while let Some(update) = StreamExt::next(&mut stream).await {
            match update {
                Ok(update) => {
                    let event_data = StreamEvent {
                        text: update.text,
                        finish_reason: update.inner.finish_reason,
                        usage: update.inner.usage,
                    };
                    if let Ok(json) = serde_json::to_string(&event_data) {
                        let event = Event::default().data(json);
                        if tx.send(Ok(event)).await.is_err() {
                            // Client disconnected.
                            break;
                        }
                    }
                }
                Err(err) => {
                    error!(error = %err, "Stream error");
                    break;
                }
            }
        }

        // Send the [DONE] sentinel.
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    let sse_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Sse::new(sse_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// POST /v1/chat/completions
async fn handle_chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    debug!(messages = request.messages.len(), "Chat completions request");

    let messages: Vec<Message> = request
        .messages
        .into_iter()
        .map(|m| m.into_message())
        .collect();

    let mut session = AgentSession::new();

    let options = if request.max_tokens.is_some() || request.temperature.is_some() || request.model.is_some() {
        Some(AgentRunOptions {
            chat_options: Some(ChatOptions {
                model: request.model,
                max_tokens: request.max_tokens,
                temperature: request.temperature,
                ..Default::default()
            }),
            additional_instructions: None,
        })
    } else {
        None
    };

    let result = state
        .agent
        .run(messages, &mut session, options.as_ref())
        .await;

    match result {
        Ok(response) => {
            let usage = response.usage.as_ref().map(|u| ChatCompletionUsage {
                prompt_tokens: u.input_tokens,
                completion_tokens: u.output_tokens,
                total_tokens: u.input_tokens + u.output_tokens,
            });

            let finish_reason = match response.finish_reason {
                Some(FinishReason::Stop) => "stop",
                Some(FinishReason::MaxTokens) => "length",
                Some(FinishReason::ToolUse) => "tool_calls",
                Some(FinishReason::ContentFilter) => "content_filter",
                None => "stop",
            };

            let completion = ChatCompletionResponse {
                id: uuid::Uuid::new_v4().to_string(),
                object: "chat.completion".to_string(),
                choices: vec![ChatCompletionChoice {
                    index: 0,
                    message: ChatCompletionMessage {
                        role: "assistant".to_string(),
                        content: response.text,
                    },
                    finish_reason: finish_reason.to_string(),
                }],
                usage,
            };

            Json(completion).into_response()
        }
        Err(err) => agent_error_to_response(err),
    }
}

// ---------------------------------------------------------------------------
// Router builder
// ---------------------------------------------------------------------------

/// Wraps a `Box<dyn Agent>` and serves it as an HTTP API.
///
/// Use [`AgentServer::router`] to get the axum [`Router`], or [`AgentServer::serve`]
/// to bind and run the server.
pub struct AgentServer {
    agent: Arc<dyn Agent>,
}

impl AgentServer {
    /// Create a new server wrapping the given agent.
    pub fn new(agent: impl Agent + 'static) -> Self {
        Self {
            agent: Arc::new(agent),
        }
    }

    /// Create a new server from a pre-wrapped `Arc<dyn Agent>`.
    pub fn from_arc(agent: Arc<dyn Agent>) -> Self {
        Self { agent }
    }

    /// Build the axum [`Router`] for this server.
    pub fn router(&self) -> Router {
        build_router(Arc::clone(&self.agent))
    }

    /// Bind and serve the agent at the given configuration.
    pub async fn serve(self, config: AgentServerConfig) -> Result<(), AgentError> {
        serve_with_arc(self.agent, config).await
    }
}

/// Build an axum [`Router`] that serves the given agent.
///
/// Routes:
/// - `POST /v1/chat/completions` -- OpenAI-compatible completions
/// - `POST /v1/agent/run` -- native agent run
/// - `POST /v1/agent/run/stream` -- native agent streaming (SSE)
/// - `GET /v1/agent/info` -- agent metadata
pub fn build_router(agent: Arc<dyn Agent>) -> Router {
    let state = AppState { agent };

    Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/agent/run", post(handle_agent_run))
        .route("/v1/agent/run/stream", post(handle_agent_run_stream))
        .route("/v1/agent/info", get(handle_agent_info))
        .with_state(state)
}

/// Convenience function to bind and serve an agent.
///
/// This creates an [`AgentServer`], builds the router, and starts listening
/// on the configured host and port. It runs until the process is interrupted.
pub async fn serve(agent: impl Agent + 'static, config: AgentServerConfig) -> Result<(), AgentError> {
    serve_with_arc(Arc::new(agent), config).await
}

async fn serve_with_arc(agent: Arc<dyn Agent>, config: AgentServerConfig) -> Result<(), AgentError> {
    let router = build_router(agent);
    let addr = format!("{}:{}", config.host, config.port);

    debug!(addr = %addr, "Starting agent server");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| AgentError::HttpError(format!("failed to bind to {addr}: {e}")))?;

    axum::serve(listener, router)
        .await
        .map_err(|e| AgentError::HttpError(format!("server error: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::types::Role;

    #[test]
    fn agent_run_request_serialization() {
        let request = AgentRunRequest {
            messages: vec![Message::user("Hello")],
            session_id: Some("session-123".to_string()),
            options: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: AgentRunRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.messages.len(), 1);
        assert_eq!(deserialized.messages[0].text(), "Hello");
        assert_eq!(deserialized.session_id, Some("session-123".to_string()));
    }

    #[test]
    fn agent_run_response_serialization() {
        let response = AgentRunResponse {
            text: "Hello!".to_string(),
            messages: vec![Message::assistant("Hello!")],
            finish_reason: Some(FinishReason::Stop),
            usage: Some(Usage {
                input_tokens: 10,
                output_tokens: 5,
            }),
        };

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: AgentRunResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.text, "Hello!");
        assert_eq!(deserialized.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn chat_completion_request_serialization() {
        let request = ChatCompletionRequest {
            messages: vec![ChatCompletionMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            model: Some("gpt-4".to_string()),
            max_tokens: Some(100),
            temperature: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        let deserialized: ChatCompletionRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.messages.len(), 1);
        assert_eq!(deserialized.model, Some("gpt-4".to_string()));
        assert_eq!(deserialized.max_tokens, Some(100));
    }

    #[test]
    fn chat_completion_response_serialization() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_string(),
                    content: "Hi there!".to_string(),
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Some(ChatCompletionUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        };

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: ChatCompletionResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, "chatcmpl-123");
        assert_eq!(deserialized.choices.len(), 1);
        assert_eq!(deserialized.choices[0].message.content, "Hi there!");
        assert_eq!(deserialized.usage.as_ref().unwrap().total_tokens, 15);
    }

    #[test]
    fn chat_completion_message_converts_to_message() {
        let system = ChatCompletionMessage {
            role: "system".to_string(),
            content: "You are helpful.".to_string(),
        };
        let msg = system.into_message();
        assert_eq!(msg.role, Role::System);
        assert_eq!(msg.text(), "You are helpful.");

        let user = ChatCompletionMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        };
        let msg = user.into_message();
        assert_eq!(msg.role, Role::User);

        let assistant = ChatCompletionMessage {
            role: "assistant".to_string(),
            content: "Hi".to_string(),
        };
        let msg = assistant.into_message();
        assert_eq!(msg.role, Role::Assistant);
    }

    #[test]
    fn stream_event_serialization() {
        let event = StreamEvent {
            text: Some("chunk".to_string()),
            finish_reason: None,
            usage: None,
        };

        let json = serde_json::to_string(&event).unwrap();
        let deserialized: StreamEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.text, Some("chunk".to_string()));
        assert!(deserialized.finish_reason.is_none());
    }

    #[test]
    fn agent_info_response_serialization() {
        let info = AgentInfoResponse {
            id: "agent-1".to_string(),
            name: Some("My Agent".to_string()),
            description: None,
        };

        let json = serde_json::to_string(&info).unwrap();
        let deserialized: AgentInfoResponse = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.id, "agent-1");
        assert_eq!(deserialized.name, Some("My Agent".to_string()));
        assert!(deserialized.description.is_none());
    }

    #[test]
    fn default_config() {
        let config = AgentServerConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
    }

    #[test]
    fn agent_run_request_options_convert() {
        let opts = AgentRunRequestOptions {
            chat_options: Some(ChatOptions {
                temperature: Some(0.5),
                ..Default::default()
            }),
            additional_instructions: Some("Be brief.".to_string()),
        };

        let run_opts: AgentRunOptions = opts.into();
        assert_eq!(run_opts.chat_options.unwrap().temperature, Some(0.5));
        assert_eq!(run_opts.additional_instructions.unwrap(), "Be brief.");
    }
}
