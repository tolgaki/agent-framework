// Copyright (c) Microsoft. All rights reserved.

//! Sequential orchestration: run agents one after another, piping output → input.

use agent_framework_core::agent::Agent;
use agent_framework_core::error::AgentResult;
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::{AgentResponse, Message};

/// Run a sequence of agents, where each agent's text output becomes the next
/// agent's user input.
///
/// Mirrors Python's `SequentialBuilder`.
pub struct SequentialOrchestration {
    agents: Vec<Box<dyn Agent>>,
}

impl SequentialOrchestration {
    pub fn new() -> Self {
        Self { agents: Vec::new() }
    }

    pub fn add_agent(mut self, agent: impl Agent + 'static) -> Self {
        self.agents.push(Box::new(agent));
        self
    }

    /// Run all agents in sequence. The initial messages are sent to the first
    /// agent; each subsequent agent receives the previous agent's text as a
    /// user message.
    pub async fn run(
        &self,
        initial_messages: Vec<Message>,
        session: &mut AgentSession,
    ) -> AgentResult<AgentResponse> {
        let mut messages = initial_messages;
        let mut last_response = None;

        for agent in &self.agents {
            let response = agent.run(messages, session, None).await?;
            messages = vec![Message::user(&response.text)];
            last_response = Some(response);
        }

        last_response.ok_or_else(|| {
            agent_framework_core::error::AgentError::InvalidRequest(
                "sequential orchestration has no agents".to_string(),
            )
        })
    }
}

impl Default for SequentialOrchestration {
    fn default() -> Self {
        Self::new()
    }
}
