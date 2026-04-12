// Copyright (c) Microsoft. All rights reserved.

//! Server-side: expose an [`Agent`] over the A2A protocol via `a2a-rs-server`.

use std::sync::Arc;

use async_trait::async_trait;
use tracing::error;

use a2a_rs_core::{
    completed_task_with_text, AgentCapabilities, AgentCard, AgentInterface, AgentProvider,
    AgentSkill, Message as A2aMessage, Part as A2aPart, SendMessageResponse,
};
use a2a_rs_server::{AuthContext, HandlerError, HandlerResult, MessageHandler};

use agent_framework_core::agent::Agent;
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::Message;

/// Wraps an [`Agent`] so it can be served as an A2A endpoint via
/// [`a2a_rs_server::A2aServer`].
///
/// # Example
///
/// ```rust,no_run
/// # use std::sync::Arc;
/// # use a2a_rs_server::A2aServer;
/// # use agent_framework_a2a::A2AAgentHandler;
/// # async fn run(agent: Arc<dyn agent_framework_core::agent::Agent>) -> anyhow::Result<()> {
/// let handler = A2AAgentHandler::new(agent);
/// A2aServer::new(handler).bind("0.0.0.0:8080")?.run().await
/// # }
/// ```
pub struct A2AAgentHandler {
    agent: Arc<dyn Agent>,
    organization: String,
    organization_url: String,
    skill: AgentSkill,
}

impl A2AAgentHandler {
    /// Wrap an [`Agent`] for A2A serving.
    pub fn new(agent: Arc<dyn Agent>) -> Self {
        let id = agent.id().to_string();
        let name = agent.name().unwrap_or("agent").to_string();
        let description = agent.description().unwrap_or("").to_string();
        Self {
            agent,
            organization: "Microsoft Agent Framework".to_string(),
            organization_url: "https://github.com/microsoft/agent-framework".to_string(),
            skill: AgentSkill {
                id,
                name,
                description,
                tags: vec!["chat".to_string()],
                ..Default::default()
            },
        }
    }

    /// Override the advertised organization name.
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = org.into();
        self
    }

    /// Override the advertised organization URL.
    pub fn with_organization_url(mut self, url: impl Into<String>) -> Self {
        self.organization_url = url.into();
        self
    }
}

#[async_trait]
impl MessageHandler for A2AAgentHandler {
    async fn handle_message(
        &self,
        message: A2aMessage,
        _auth: Option<AuthContext>,
    ) -> HandlerResult<SendMessageResponse> {
        // Extract text from the inbound A2A message.
        let user_text: String = message
            .parts
            .iter()
            .filter_map(part_text)
            .collect::<Vec<_>>()
            .join("\n");

        // Run the agent with a fresh session for each message. Cross-message
        // continuity should be threaded through `context_id`/`task_id` if the
        // caller wants conversation state — A2A is intentionally stateless on
        // the wire.
        let mut session = AgentSession::new();
        let inbound = vec![Message::user(user_text)];

        let response = self
            .agent
            .run(inbound, &mut session, None)
            .await
            .map_err(|e| {
                error!(error = %e, "agent.run failed inside A2A handler");
                HandlerError::processing_failed(format!("agent run failed: {e}"))
            })?;

        let reply_text = response.text;
        Ok(SendMessageResponse::Task(completed_task_with_text(
            message,
            &reply_text,
        )))
    }

    fn agent_card(&self, base_url: &str) -> AgentCard {
        let agent_name = self.agent.name().unwrap_or("Agent").to_string();
        let agent_desc = self
            .agent
            .description()
            .unwrap_or("Microsoft Agent Framework agent")
            .to_string();

        AgentCard {
            name: agent_name,
            description: agent_desc,
            supported_interfaces: vec![AgentInterface {
                url: format!("{}/v1/rpc", base_url.trim_end_matches('/')),
                protocol_binding: "JSONRPC".to_string(),
                protocol_version: a2a_rs_core::PROTOCOL_VERSION.to_string(),
                tenant: None,
            }],
            provider: Some(AgentProvider {
                organization: self.organization.clone(),
                url: self.organization_url.clone(),
            }),
            version: a2a_rs_core::PROTOCOL_VERSION.to_string(),
            capabilities: AgentCapabilities {
                streaming: Some(false),
                push_notifications: Some(false),
                extended_agent_card: Some(false),
                extensions: vec![],
            },
            skills: vec![self.skill.clone()],
            ..Default::default()
        }
    }
}

/// Helper: pull text out of an A2A `Part::Text` variant.
fn part_text(part: &A2aPart) -> Option<String> {
    match part {
        A2aPart::Text { text, .. } => Some(text.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use a2a_rs_core::Role;
    use agent_framework_core::agent::ChatClientAgent;
    use agent_framework_core::client::ChatClient;
    use agent_framework_core::error::AgentResult;
    use agent_framework_core::streaming::ResponseStream;
    use agent_framework_core::types::{ChatOptions, ChatResponse, FinishReason};

    struct StubClient;

    #[async_trait]
    impl ChatClient for StubClient {
        async fn get_response(
            &self,
            _: &[Message],
            _: Option<&ChatOptions>,
        ) -> AgentResult<ChatResponse> {
            Ok(ChatResponse {
                messages: vec![Message::assistant("hello from stub")],
                response_id: None,
                finish_reason: Some(FinishReason::Stop),
                usage: None,
            })
        }
        async fn get_response_stream(
            &self,
            _: &[Message],
            _: Option<&ChatOptions>,
        ) -> AgentResult<ResponseStream> {
            Err(agent_framework_core::error::AgentError::Unimplemented(
                "stream".into(),
            ))
        }
    }

    fn make_handler() -> A2AAgentHandler {
        let agent = ChatClientAgent::builder()
            .client(StubClient)
            .name("test-agent")
            .instructions("be helpful")
            .build()
            .unwrap();
        A2AAgentHandler::new(Arc::new(agent))
    }

    #[tokio::test]
    async fn handler_runs_agent_and_returns_task_with_reply_text() {
        let handler = make_handler();
        let inbound = A2aMessage {
            kind: "message".to_string(),
            message_id: "m1".to_string(),
            context_id: None,
            task_id: None,
            role: Role::User,
            parts: vec![A2aPart::Text {
                text: "hi".to_string(),
                metadata: None,
            }],
            extensions: vec![],
            reference_task_ids: None,
            metadata: None,
        };

        let result = handler.handle_message(inbound, None).await.unwrap();
        match result {
            SendMessageResponse::Task(task) => {
                let history = task.history.unwrap_or_default();
                let collected: String = history
                    .iter()
                    .flat_map(|m| m.parts.iter().filter_map(part_text))
                    .collect::<Vec<_>>()
                    .join(" ");
                assert!(
                    collected.contains("hello from stub"),
                    "expected stub reply in task history, got {collected:?}"
                );
            }
            SendMessageResponse::Message(_) => panic!("expected Task response"),
        }
    }

    #[test]
    fn agent_card_advertises_jsonrpc_interface() {
        let handler = make_handler();
        let card = handler.agent_card("https://example.com");
        assert_eq!(card.supported_interfaces.len(), 1);
        assert!(card.supported_interfaces[0].url.ends_with("/v1/rpc"));
        assert_eq!(card.supported_interfaces[0].protocol_binding, "JSONRPC");
        assert!(card.provider.is_some());
    }
}
