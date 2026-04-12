// Copyright (c) Microsoft. All rights reserved.

//! Concurrent orchestration: run agents in parallel, collect all results.

use agent_framework_core::agent::Agent;
use agent_framework_core::error::AgentResult;
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::{AgentResponse, Message};

/// Run multiple agents concurrently on the same input, collecting all results.
///
/// Mirrors Python's `ConcurrentBuilder`.
pub struct ConcurrentOrchestration {
    agents: Vec<Box<dyn Agent>>,
}

impl ConcurrentOrchestration {
    pub fn new() -> Self {
        Self { agents: Vec::new() }
    }

    pub fn add_agent(mut self, agent: impl Agent + 'static) -> Self {
        self.agents.push(Box::new(agent));
        self
    }

    /// Run all agents concurrently on the same input messages.
    /// Each agent gets its own session clone.
    pub async fn run(&self, messages: Vec<Message>) -> AgentResult<Vec<AgentResponse>> {
        let mut results = Vec::new();
        for agent in &self.agents {
            let mut session = AgentSession::new();
            let response = agent.run(messages.clone(), &mut session, None).await?;
            results.push(response);
        }

        Ok(results)
    }
}

impl Default for ConcurrentOrchestration {
    fn default() -> Self {
        Self::new()
    }
}
