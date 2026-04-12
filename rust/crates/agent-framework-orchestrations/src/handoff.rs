// Copyright (c) Microsoft. All rights reserved.

//! Handoff orchestration: agent-to-agent handoff via tool calls.

use std::collections::HashMap;

use agent_framework_core::agent::Agent;
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::{AgentResponse, Content, Message};

/// A target agent that can be handed off to.
pub struct HandoffTarget {
    /// The agent to hand off to.
    pub agent: Box<dyn Agent>,
    /// The tool name that triggers handoff to this agent.
    /// When the current agent calls this tool, control transfers.
    pub tool_name: String,
    /// Description shown to the model.
    pub description: String,
}

/// Multi-agent handoff orchestration.
///
/// An initial agent runs and can "hand off" to other agents by calling
/// specific tool names. The conversation continues with the target agent.
///
/// Mirrors .NET's `HandoffWorkflowBuilder` and Python's `HandoffBuilder`.
pub struct HandoffOrchestration {
    /// The starting agent.
    initial_agent: Box<dyn Agent>,
    /// Map of tool_name → HandoffTarget.
    targets: HashMap<String, HandoffTarget>,
    /// Maximum total handoffs before stopping.
    max_handoffs: usize,
}

impl HandoffOrchestration {
    pub fn new(initial_agent: impl Agent + 'static) -> Self {
        Self {
            initial_agent: Box::new(initial_agent),
            targets: HashMap::new(),
            max_handoffs: 10,
        }
    }

    /// Register a handoff target.
    pub fn add_target(mut self, target: HandoffTarget) -> Self {
        self.targets.insert(target.tool_name.clone(), target);
        self
    }

    pub fn max_handoffs(mut self, max: usize) -> Self {
        self.max_handoffs = max;
        self
    }

    /// Run the orchestration.
    pub async fn run(
        &self,
        messages: Vec<Message>,
        session: &mut AgentSession,
    ) -> AgentResult<AgentResponse> {
        let mut current_agent: &dyn Agent = self.initial_agent.as_ref();
        let mut current_messages = messages;
        let mut all_responses = Vec::new();

        for _handoff in 0..self.max_handoffs {
            let response = current_agent.run(current_messages.clone(), session, None).await?;

            // Check if the response contains a handoff tool call.
            let handoff_call = response
                .messages
                .iter()
                .flat_map(|m| m.content.iter())
                .find_map(|c| {
                    if let Content::ToolCall { name, .. } = c {
                        self.targets.get(name.as_str()).map(|_| name.clone())
                    } else {
                        None
                    }
                });

            all_responses.push(response.clone());

            match handoff_call {
                Some(tool_name) => {
                    let target = self.targets.get(&tool_name).unwrap();
                    current_agent = target.agent.as_ref();
                    // Continue conversation: pass the accumulated messages
                    // including the handoff request.
                    current_messages = vec![Message::user(&response.text)];
                }
                None => {
                    // No handoff — return the final response.
                    return Ok(response);
                }
            }
        }

        // Exceeded max handoffs — return the last response.
        all_responses
            .pop()
            .ok_or_else(|| AgentError::InvalidResponse("no responses generated".to_string()))
    }
}
