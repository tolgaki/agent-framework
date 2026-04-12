// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework AG-UI
//!
//! [AG-UI (Agentic UI)](https://docs.ag-ui.com) protocol support for the
//! Microsoft Agent Framework. Wraps any [`Agent`] and exposes it via
//! Server-Sent Events using the AG-UI event format.

use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::routing::post;
use axum::Json;
use axum::Router;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tracing::warn;
use uuid::Uuid;

use agent_framework_core::agent::Agent;
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::Message;

// ---------------------------------------------------------------------------
// AG-UI event types
// ---------------------------------------------------------------------------

/// AG-UI protocol events sent over Server-Sent Events.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AguiEvent {
    /// Signals the start of an agent run.
    RunStarted {
        thread_id: String,
        run_id: String,
    },

    /// Signals the start of a new text message from the agent.
    TextMessageStart {
        message_id: String,
        role: String,
    },

    /// An incremental text content delta.
    TextMessageContent {
        message_id: String,
        delta: String,
    },

    /// Signals the end of a text message.
    TextMessageEnd {
        message_id: String,
    },

    /// Signals the start of a tool call.
    ToolCallStart {
        tool_call_id: String,
        tool_name: String,
    },

    /// An incremental tool call argument delta.
    ToolCallArgs {
        tool_call_id: String,
        delta: String,
    },

    /// Signals the end of a tool call.
    ToolCallEnd {
        tool_call_id: String,
    },

    /// Signals the end of an agent run.
    RunFinished {
        thread_id: String,
        run_id: String,
    },
}

impl AguiEvent {
    /// Return the SSE event name for this event type.
    pub fn event_name(&self) -> &'static str {
        match self {
            Self::RunStarted { .. } => "RUN_STARTED",
            Self::TextMessageStart { .. } => "TEXT_MESSAGE_START",
            Self::TextMessageContent { .. } => "TEXT_MESSAGE_CONTENT",
            Self::TextMessageEnd { .. } => "TEXT_MESSAGE_END",
            Self::ToolCallStart { .. } => "TOOL_CALL_START",
            Self::ToolCallArgs { .. } => "TOOL_CALL_ARGS",
            Self::ToolCallEnd { .. } => "TOOL_CALL_END",
            Self::RunFinished { .. } => "RUN_FINISHED",
        }
    }
}

// ---------------------------------------------------------------------------
// AG-UI request types
// ---------------------------------------------------------------------------

/// An AG-UI message in the input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AguiMessage {
    /// The role of the message author.
    pub role: String,
    /// The text content.
    pub content: String,
}

/// An AG-UI tool definition in the input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AguiTool {
    /// The tool name.
    pub name: String,
    /// A description of the tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the tool parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// Input for an AG-UI run request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunAgentInput {
    /// The thread (conversation) identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    /// The conversation messages.
    pub messages: Vec<AguiMessage>,
    /// Optional tool definitions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AguiTool>,
}

// ---------------------------------------------------------------------------
// Router builder
// ---------------------------------------------------------------------------

/// Build an Axum router that serves the AG-UI protocol for the given agent.
///
/// The router exposes a single `POST /` endpoint that accepts
/// [`RunAgentInput`] and streams [`AguiEvent`]s via Server-Sent Events.
pub fn build_agui_router(agent: Arc<dyn Agent>) -> Router {
    Router::new()
        .route("/", post(agui_handler))
        .with_state(agent)
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

async fn agui_handler(
    State(agent): State<Arc<dyn Agent>>,
    Json(input): Json<RunAgentInput>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let thread_id = input.thread_id.unwrap_or_else(|| Uuid::new_v4().to_string());
    let run_id = Uuid::new_v4().to_string();
    let message_id = Uuid::new_v4().to_string();

    let messages: Vec<Message> = input
        .messages
        .iter()
        .map(|m| match m.role.as_str() {
            "system" => Message::system(&m.content),
            "assistant" => Message::assistant(&m.content),
            _ => Message::user(&m.content),
        })
        .collect();

    // Use a channel to bridge the non-'static agent stream into SSE.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);

    tokio::spawn(async move {
        let send = |event: AguiEvent| {
            let tx = tx.clone();
            async move {
                let _ = tx.send(Ok(sse_event(&event))).await;
            }
        };

        send(AguiEvent::RunStarted {
            thread_id: thread_id.clone(),
            run_id: run_id.clone(),
        })
        .await;

        let mut session = AgentSession::new();

        match agent.run_stream(messages.clone(), &mut session, None) {
            Ok(mut stream) => {
                use std::pin::Pin;

                send(AguiEvent::TextMessageStart {
                    message_id: message_id.clone(),
                    role: "assistant".to_string(),
                })
                .await;

                while let Some(result) = Pin::new(&mut stream).next().await {
                    match result {
                        Ok(update) => {
                            if let Some(text) = update.text {
                                send(AguiEvent::TextMessageContent {
                                    message_id: message_id.clone(),
                                    delta: text,
                                })
                                .await;
                            }
                        }
                        Err(e) => {
                            warn!(error = %e, "Stream error during AG-UI run");
                        }
                    }
                }

                send(AguiEvent::TextMessageEnd {
                    message_id: message_id.clone(),
                })
                .await;
            }
            Err(_) => {
                // Fallback: run synchronously and emit all events at once.
                let mut session = AgentSession::new();
                match agent.run(messages, &mut session, None).await {
                    Ok(response) => {
                        send(AguiEvent::TextMessageStart {
                            message_id: message_id.clone(),
                            role: "assistant".to_string(),
                        })
                        .await;

                        if !response.text.is_empty() {
                            send(AguiEvent::TextMessageContent {
                                message_id: message_id.clone(),
                                delta: response.text,
                            })
                            .await;
                        }

                        // Emit tool calls from the response messages.
                        for msg in &response.messages {
                            for (tc_id, tc_name, tc_args) in msg.tool_calls() {
                                send(AguiEvent::ToolCallStart {
                                    tool_call_id: tc_id.to_string(),
                                    tool_name: tc_name.to_string(),
                                })
                                .await;
                                send(AguiEvent::ToolCallArgs {
                                    tool_call_id: tc_id.to_string(),
                                    delta: tc_args.to_string(),
                                })
                                .await;
                                send(AguiEvent::ToolCallEnd {
                                    tool_call_id: tc_id.to_string(),
                                })
                                .await;
                            }
                        }

                        send(AguiEvent::TextMessageEnd {
                            message_id: message_id.clone(),
                        })
                        .await;
                    }
                    Err(e) => {
                        warn!(error = %e, "Agent run failed during AG-UI request");
                    }
                }
            }
        }

        send(AguiEvent::RunFinished {
            thread_id,
            run_id,
        })
        .await;
    });

    let sse_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Sse::new(sse_stream)
}

fn sse_event(event: &AguiEvent) -> Event {
    let data = serde_json::to_string(event).unwrap_or_default();
    Event::default().event(event.event_name()).data(data)
}
