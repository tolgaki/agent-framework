// Copyright (c) Microsoft. All rights reserved.

//! Client-side: consume a remote A2A agent as if it were a local [`Agent`].

use async_trait::async_trait;
use uuid::Uuid;

use a2a_rs_client::{A2aClient, ClientConfig};
use a2a_rs_core::{Message as A2aMessage, Part as A2aPart, Role as A2aRole, SendMessageResult, Task};

use agent_framework_core::agent::Agent;
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::session::AgentSession;
use agent_framework_core::streaming::AgentResponseStream;
use agent_framework_core::types::{
    AgentResponse, AgentResponseUpdate, AgentRunOptions, ChatResponseUpdate, FinishReason, Message,
    Role,
};

/// Configuration for an [`A2ARemoteAgent`].
#[derive(Debug, Clone)]
pub struct A2AClientConfig {
    /// Base URL of the remote A2A server (e.g. `http://localhost:8080`).
    pub server_url: String,
    /// Optional pre-issued OAuth/session token forwarded with each request.
    pub session_token: Option<String>,
}

impl A2AClientConfig {
    pub fn new(server_url: impl Into<String>) -> Self {
        Self {
            server_url: server_url.into(),
            session_token: None,
        }
    }

    pub fn with_session_token(mut self, token: impl Into<String>) -> Self {
        self.session_token = Some(token.into());
        self
    }
}

/// An [`Agent`] backed by a remote A2A endpoint.
///
/// Wraps `a2a-rs-client::A2aClient` and translates framework messages into A2A
/// `Message`s on the way out, then converts the returned `SendMessageResult`
/// back into framework types.
pub struct A2ARemoteAgent {
    id: String,
    name: Option<String>,
    description: Option<String>,
    client: A2aClient,
    session_token: Option<String>,
}

impl A2ARemoteAgent {
    /// Create a remote agent from a [`A2AClientConfig`].
    pub fn new(config: A2AClientConfig) -> AgentResult<Self> {
        let client_config = ClientConfig {
            server_url: config.server_url.clone(),
            ..Default::default()
        };
        let client = A2aClient::new(client_config)
            .map_err(|e| AgentError::InvalidRequest(format!("failed to build A2A client: {e}")))?;
        Ok(Self {
            id: Uuid::new_v4().to_string(),
            name: None,
            description: None,
            client,
            session_token: config.session_token,
        })
    }

    /// Convenience: create a remote agent and fetch its agent card to populate
    /// `name` / `description`. Use this when you want the local proxy to mirror
    /// the remote's identity.
    pub async fn connect(server_url: impl Into<String>) -> AgentResult<Self> {
        let config = A2AClientConfig::new(server_url);
        let mut agent = Self::new(config)?;
        agent.refresh_card().await?;
        Ok(agent)
    }

    /// Fetch the remote agent card and update local `name` / `description`.
    pub async fn refresh_card(&mut self) -> AgentResult<()> {
        let card = self
            .client
            .fetch_agent_card()
            .await
            .map_err(|e| AgentError::HttpError(format!("failed to fetch agent card: {e}")))?;
        self.name = Some(card.name);
        if !card.description.is_empty() {
            self.description = Some(card.description);
        }
        Ok(())
    }
}

#[async_trait]
impl Agent for A2ARemoteAgent {
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
        _session: &mut AgentSession,
        _options: Option<&AgentRunOptions>,
    ) -> AgentResult<AgentResponse> {
        // Concatenate the user-visible portion of the input into a single A2A
        // message. A2A's wire shape is one message per call, so multi-turn
        // history is the responsibility of either `_session` (locally) or
        // `context_id` on the remote side.
        let combined_text = messages
            .iter()
            .filter(|m| m.role == Role::User || m.role == Role::System)
            .map(|m| m.text())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        let outbound = A2aMessage {
            kind: "message".to_string(),
            message_id: Uuid::new_v4().to_string(),
            context_id: None,
            task_id: None,
            role: A2aRole::User,
            parts: vec![A2aPart::Text {
                text: combined_text,
                metadata: None,
            }],
            extensions: vec![],
            reference_task_ids: None,
            metadata: None,
        };

        let result = self
            .client
            .send_message(outbound, self.session_token.as_deref(), None)
            .await
            .map_err(|e| AgentError::HttpError(format!("A2A send_message failed: {e}")))?;

        Ok(send_result_to_agent_response(result))
    }

    fn run_stream<'a>(
        &'a self,
        messages: Vec<Message>,
        session: &'a mut AgentSession,
        options: Option<&'a AgentRunOptions>,
    ) -> AgentResult<AgentResponseStream<'a>> {
        // Wrap the synchronous `run()` result as a single-update stream so the
        // contract matches local agents. Native streaming via
        // `send_message_streaming` could be added later as an optimization.
        let stream = async_stream_from_run(self, messages, session, options);
        Ok(AgentResponseStream::new(stream))
    }
}

/// Build a single-update stream from a normal `run` call. Used by `run_stream`
/// for parity with local agents that always produce a stream.
fn async_stream_from_run<'a>(
    agent: &'a A2ARemoteAgent,
    messages: Vec<Message>,
    session: &'a mut AgentSession,
    options: Option<&'a AgentRunOptions>,
) -> impl futures_util_stream::Stream<Item = AgentResult<AgentResponseUpdate>> + Send + 'a {
    use futures_util_stream::stream::once;

    once(async move {
        let response = agent.run(messages, session, options).await?;
        Ok(AgentResponseUpdate {
            text: Some(response.text.clone()),
            inner: ChatResponseUpdate {
                text: Some(response.text),
                tool_call: None,
                finish_reason: response.finish_reason,
                usage: response.usage,
            },
        })
    })
}

// We re-export the stream traits we need from `futures` (already a workspace
// dep) under a local alias so the function signatures above stay terse.
mod futures_util_stream {
    pub use futures::stream;
    pub use futures::Stream;
}

/// Convert an A2A `SendMessageResult` into a framework [`AgentResponse`].
fn send_result_to_agent_response(result: SendMessageResult) -> AgentResponse {
    match result {
        SendMessageResult::Message(msg) => message_to_agent_response(msg),
        SendMessageResult::Task(task) => task_to_agent_response(task),
    }
}

fn message_to_agent_response(msg: A2aMessage) -> AgentResponse {
    let text = collect_text(&msg.parts);
    AgentResponse {
        messages: vec![Message::assistant(text.clone())],
        text,
        finish_reason: Some(FinishReason::Stop),
        usage: None,
    }
}

fn task_to_agent_response(task: Task) -> AgentResponse {
    // Concatenate text from every assistant message in task history, then fall
    // back to artifact text if history is empty (some servers only emit
    // artifacts).
    let history = task.history.unwrap_or_default();
    let artifacts = task.artifacts.unwrap_or_default();

    let mut text_parts: Vec<String> = history
        .iter()
        .filter(|m| matches!(m.role, A2aRole::Agent))
        .map(|m| collect_text(&m.parts))
        .filter(|t| !t.is_empty())
        .collect();

    if text_parts.is_empty() {
        for artifact in &artifacts {
            let t = collect_text(&artifact.parts);
            if !t.is_empty() {
                text_parts.push(t);
            }
        }
    }

    let text = text_parts.join("\n");
    let messages: Vec<Message> = history
        .iter()
        .map(|m| {
            let role = match m.role {
                A2aRole::User => Role::User,
                A2aRole::Agent => Role::Assistant,
                _ => Role::Assistant,
            };
            Message {
                role,
                content: vec![agent_framework_core::types::Content::text(collect_text(&m.parts))],
                name: None,
                metadata: Default::default(),
            }
        })
        .collect();

    AgentResponse {
        messages,
        text,
        finish_reason: Some(FinishReason::Stop),
        usage: None,
    }
}

fn collect_text(parts: &[A2aPart]) -> String {
    parts
        .iter()
        .filter_map(|p| match p {
            A2aPart::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;
    use a2a_rs_core::{TaskState, TaskStatus};

    fn make_text_part(s: &str) -> A2aPart {
        A2aPart::Text {
            text: s.to_string(),
            metadata: None,
        }
    }

    #[test]
    fn message_result_extracts_text() {
        let msg = A2aMessage {
            kind: "message".to_string(),
            message_id: "m".to_string(),
            context_id: None,
            task_id: None,
            role: A2aRole::Agent,
            parts: vec![make_text_part("hello world")],
            extensions: vec![],
            reference_task_ids: None,
            metadata: None,
        };
        let resp = send_result_to_agent_response(SendMessageResult::Message(msg));
        assert_eq!(resp.text, "hello world");
        assert_eq!(resp.finish_reason, Some(FinishReason::Stop));
    }

    #[test]
    fn task_result_concatenates_agent_history() {
        let task = Task {
            id: "t1".into(),
            context_id: "c1".into(),
            kind: "task".to_string(),
            status: TaskStatus {
                state: TaskState::Completed,
                message: None,
                timestamp: None,
            },
            history: Some(vec![
                A2aMessage {
                    kind: "message".to_string(),
                    message_id: "u1".into(),
                    context_id: None,
                    task_id: Some("t1".into()),
                    role: A2aRole::User,
                    parts: vec![make_text_part("question")],
                    extensions: vec![],
                    reference_task_ids: None,
                    metadata: None,
                },
                A2aMessage {
                    kind: "message".to_string(),
                    message_id: "a1".into(),
                    context_id: None,
                    task_id: Some("t1".into()),
                    role: A2aRole::Agent,
                    parts: vec![make_text_part("part1 ")],
                    extensions: vec![],
                    reference_task_ids: None,
                    metadata: None,
                },
                A2aMessage {
                    kind: "message".to_string(),
                    message_id: "a2".into(),
                    context_id: None,
                    task_id: Some("t1".into()),
                    role: A2aRole::Agent,
                    parts: vec![make_text_part("part2")],
                    extensions: vec![],
                    reference_task_ids: None,
                    metadata: None,
                },
            ]),
            artifacts: None,
            metadata: None,
        };

        let resp = send_result_to_agent_response(SendMessageResult::Task(task));
        assert_eq!(resp.text, "part1 \npart2");
        assert_eq!(resp.messages.len(), 3);
    }

    #[test]
    fn config_supports_session_token() {
        let cfg = A2AClientConfig::new("http://localhost:8080").with_session_token("tok");
        assert_eq!(cfg.server_url, "http://localhost:8080");
        assert_eq!(cfg.session_token.as_deref(), Some("tok"));
    }
}
