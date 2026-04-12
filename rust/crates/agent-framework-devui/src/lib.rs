// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework Dev UI
//!
//! A development UI server that wraps one or more agents and exposes them
//! through a simple web interface for testing and debugging.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{Html, Json};
use axum::routing::{get, post};
use axum::Router;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};

use agent_framework_core::agent::Agent;
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::Message;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the development UI server.
#[derive(Debug, Clone)]
pub struct DevServerConfig {
    /// The host address to bind to.
    pub host: String,
    /// The port to listen on.
    pub port: u16,
    /// The title displayed in the web UI.
    pub title: String,
}

impl Default for DevServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            title: "Agent Framework Dev UI".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

/// Information about an available agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    /// The agent's unique identifier.
    pub id: String,
    /// The agent's human-readable name.
    pub name: Option<String>,
    /// A description of the agent's purpose.
    pub description: Option<String>,
}

/// A message in a run request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRequestMessage {
    /// The role of the message author.
    pub role: String,
    /// The text content of the message.
    pub content: String,
}

/// Request body for running an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRequest {
    /// The conversation messages to send.
    pub messages: Vec<RunRequestMessage>,
}

/// Response from a non-streaming agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResponse {
    /// The agent's text response.
    pub text: String,
    /// All messages produced during the run.
    pub messages: Vec<RunResponseMessage>,
}

/// A message in a run response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResponseMessage {
    /// The role of the message author.
    pub role: String,
    /// The text content of the message.
    pub content: String,
}

/// Error response body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    /// A human-readable error message.
    pub error: String,
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

struct AppState {
    agents: HashMap<String, Arc<dyn Agent>>,
    config: DevServerConfig,
}

// ---------------------------------------------------------------------------
// DevServer builder
// ---------------------------------------------------------------------------

/// A development UI server that wraps one or more agents.
///
/// # Example
///
/// ```rust,no_run
/// use agent_framework_devui::DevServer;
/// # use std::sync::Arc;
/// # fn example(agent: Arc<dyn agent_framework_core::agent::Agent>) {
/// let server = DevServer::new()
///     .add_agent(agent);
/// // server.serve("127.0.0.1", 3000).await.unwrap();
/// # }
/// ```
pub struct DevServer {
    agents: HashMap<String, Arc<dyn Agent>>,
    config: DevServerConfig,
}

impl DevServer {
    /// Create a new empty dev server with default configuration.
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            config: DevServerConfig::default(),
        }
    }

    /// Create a new dev server with the given configuration.
    pub fn with_config(config: DevServerConfig) -> Self {
        Self {
            agents: HashMap::new(),
            config,
        }
    }

    /// Add an agent to the server. Returns `self` for chaining.
    pub fn add_agent(mut self, agent: Arc<dyn Agent>) -> Self {
        let id = agent.id().to_string();
        self.agents.insert(id, agent);
        self
    }

    /// Start serving on the given host and port.
    pub async fn serve(self, host: &str, port: u16) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let config = DevServerConfig {
            host: host.to_string(),
            port,
            ..self.config
        };

        let state = Arc::new(AppState {
            agents: self.agents,
            config: config.clone(),
        });

        let app = build_router(state);
        let addr = format!("{}:{}", config.host, config.port);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        info!("Dev UI server listening on http://{}", addr);
        axum::serve(listener, app).await?;
        Ok(())
    }
}

impl Default for DevServer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/api/agents", get(list_agents_handler))
        .route("/api/agents/{id}/run", post(run_agent_handler))
        .route("/api/agents/{id}/stream", post(stream_agent_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn index_handler(State(state): State<Arc<AppState>>) -> Html<String> {
    Html(build_index_html(&state.config.title))
}

async fn list_agents_handler(State(state): State<Arc<AppState>>) -> Json<Vec<AgentInfo>> {
    let agents: Vec<AgentInfo> = state
        .agents
        .iter()
        .map(|(id, agent)| AgentInfo {
            id: id.clone(),
            name: agent.name().map(|s| s.to_string()),
            description: agent.description().map(|s| s.to_string()),
        })
        .collect();
    Json(agents)
}

async fn run_agent_handler(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(request): Json<RunRequest>,
) -> Result<Json<RunResponse>, (StatusCode, Json<ErrorResponse>)> {
    let agent = state.agents.get(&id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Agent '{}' not found", id),
            }),
        )
    })?;

    let messages = convert_messages(&request.messages);
    let mut session = AgentSession::new();

    let response = agent.run(messages, &mut session, None).await.map_err(|e| {
        warn!(agent_id = %id, error = %e, "Agent run failed");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let response_messages = response
        .messages
        .iter()
        .map(|m| RunResponseMessage {
            role: format!("{:?}", m.role).to_lowercase(),
            content: m.text(),
        })
        .collect();

    Ok(Json(RunResponse {
        text: response.text,
        messages: response_messages,
    }))
}

async fn stream_agent_handler(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(request): Json<RunRequest>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>>, (StatusCode, Json<ErrorResponse>)>
{
    let agent = Arc::clone(state.agents.get(&id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: format!("Agent '{}' not found", id),
            }),
        )
    })?);

    let messages = convert_messages(&request.messages);

    // Use a channel to bridge the non-'static run_stream into an SSE-compatible
    // 'static stream. A spawned task drives the agent stream and forwards updates.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(32);

    tokio::spawn(async move {
        let mut session = AgentSession::new();
        {
            let stream_result = agent.run_stream(messages, &mut session, None);
            match stream_result {
                Ok(mut stream) => {
                    use std::pin::Pin;

                    while let Some(result) = Pin::new(&mut stream).next().await {
                        let event = match result {
                            Ok(update) => {
                                let data = serde_json::to_string(&update).unwrap_or_default();
                                Event::default().data(data)
                            }
                            Err(e) => Event::default().event("error").data(e.to_string()),
                        };
                        if tx.send(Ok(event)).await.is_err() {
                            break; // Client disconnected.
                        }
                    }
                }
                Err(e) => {
                    let _ = tx
                        .send(Ok(Event::default().event("error").data(e.to_string())))
                        .await;
                }
            };
        }
    });

    let sse_stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Ok(Sse::new(sse_stream))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn convert_messages(messages: &[RunRequestMessage]) -> Vec<Message> {
    messages
        .iter()
        .map(|m| match m.role.as_str() {
            "system" => Message::system(&m.content),
            "assistant" => Message::assistant(&m.content),
            _ => Message::user(&m.content),
        })
        .collect()
}

fn build_index_html(title: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f5f5f5; height: 100vh; display: flex; flex-direction: column; }}
  header {{ background: #0078d4; color: white; padding: 16px 24px; }}
  header h1 {{ font-size: 1.2em; font-weight: 600; }}
  .container {{ flex: 1; display: flex; overflow: hidden; }}
  .sidebar {{ width: 250px; background: white; border-right: 1px solid #ddd; padding: 16px; overflow-y: auto; }}
  .sidebar h2 {{ font-size: 0.9em; text-transform: uppercase; color: #666; margin-bottom: 12px; }}
  .agent-btn {{ display: block; width: 100%; padding: 10px 12px; margin-bottom: 8px; border: 1px solid #ddd; border-radius: 6px; background: white; cursor: pointer; text-align: left; }}
  .agent-btn:hover {{ background: #f0f0f0; }}
  .agent-btn.active {{ border-color: #0078d4; background: #e8f4fd; }}
  .agent-name {{ font-weight: 600; font-size: 0.95em; }}
  .agent-desc {{ font-size: 0.8em; color: #666; margin-top: 4px; }}
  .chat {{ flex: 1; display: flex; flex-direction: column; }}
  .messages {{ flex: 1; overflow-y: auto; padding: 24px; }}
  .message {{ margin-bottom: 16px; max-width: 80%; }}
  .message.user {{ margin-left: auto; }}
  .message .bubble {{ padding: 10px 14px; border-radius: 12px; font-size: 0.95em; line-height: 1.5; }}
  .message.user .bubble {{ background: #0078d4; color: white; }}
  .message.assistant .bubble {{ background: white; border: 1px solid #ddd; }}
  .input-area {{ padding: 16px 24px; background: white; border-top: 1px solid #ddd; display: flex; gap: 8px; }}
  .input-area input {{ flex: 1; padding: 10px 14px; border: 1px solid #ddd; border-radius: 8px; font-size: 0.95em; }}
  .input-area button {{ padding: 10px 20px; background: #0078d4; color: white; border: none; border-radius: 8px; cursor: pointer; font-size: 0.95em; }}
  .input-area button:disabled {{ background: #ccc; }}
  .placeholder {{ display: flex; align-items: center; justify-content: center; flex: 1; color: #999; }}
</style>
</head>
<body>
<header><h1>{title}</h1></header>
<div class="container">
  <div class="sidebar">
    <h2>Agents</h2>
    <div id="agent-list"></div>
  </div>
  <div class="chat" id="chat">
    <div class="placeholder">Select an agent to start chatting</div>
  </div>
</div>
<script>
let currentAgent = null;
const agentList = document.getElementById('agent-list');
const chat = document.getElementById('chat');

async function loadAgents() {{
  const res = await fetch('/api/agents');
  const agents = await res.json();
  agentList.innerHTML = '';
  agents.forEach(a => {{
    const btn = document.createElement('button');
    btn.className = 'agent-btn';
    btn.innerHTML = '<div class="agent-name">' + (a.name || a.id) + '</div>' +
      (a.description ? '<div class="agent-desc">' + a.description + '</div>' : '');
    btn.onclick = () => selectAgent(a, btn);
    agentList.appendChild(btn);
  }});
}}

function selectAgent(agent, btn) {{
  currentAgent = agent;
  document.querySelectorAll('.agent-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  chat.innerHTML = '<div class="messages" id="messages"></div>' +
    '<div class="input-area"><input id="input" placeholder="Type a message..." />' +
    '<button id="send" onclick="sendMessage()">Send</button></div>';
  document.getElementById('input').addEventListener('keydown', e => {{
    if (e.key === 'Enter') sendMessage();
  }});
}}

function addMessage(role, text) {{
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'message ' + role;
  div.innerHTML = '<div class="bubble">' + text.replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</div>';
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}}

async function sendMessage() {{
  if (!currentAgent) return;
  const input = document.getElementById('input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMessage('user', text);
  const sendBtn = document.getElementById('send');
  sendBtn.disabled = true;
  try {{
    const res = await fetch('/api/agents/' + currentAgent.id + '/run', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ messages: [{{ role: 'user', content: text }}] }})
    }});
    const data = await res.json();
    if (data.error) {{ addMessage('assistant', 'Error: ' + data.error); }}
    else {{ addMessage('assistant', data.text); }}
  }} catch (e) {{
    addMessage('assistant', 'Error: ' + e.message);
  }}
  sendBtn.disabled = false;
  input.focus();
}}

loadAgents();
</script>
</body>
</html>"#,
        title = title
    )
}
