// Copyright (c) Microsoft. All rights reserved.

//! Group chat orchestration: multi-agent conversation with speaker selection.

use async_trait::async_trait;

use agent_framework_core::agent::Agent;
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::{AgentResponse, Message};

/// A participant in a group chat.
pub struct GroupChatParticipant {
    /// The agent.
    pub agent: Box<dyn Agent>,
    /// Display name used in the conversation transcript.
    pub display_name: String,
}

/// Strategy for selecting the next speaker in a group chat.
#[async_trait]
pub trait SpeakerSelector: Send + Sync {
    /// Select the next speaker given the conversation history and available
    /// participants. Returns the index into the participants list.
    async fn select_next(
        &self,
        history: &[Message],
        participants: &[GroupChatParticipant],
        current_round: usize,
    ) -> AgentResult<usize>;
}

/// Round-robin speaker selection.
///
/// Mirrors .NET's `RoundRobinGroupChatManager`.
pub struct RoundRobinSelector;

#[async_trait]
impl SpeakerSelector for RoundRobinSelector {
    async fn select_next(
        &self,
        _history: &[Message],
        participants: &[GroupChatParticipant],
        current_round: usize,
    ) -> AgentResult<usize> {
        Ok(current_round % participants.len())
    }
}

/// Termination condition for the group chat.
#[async_trait]
pub trait TerminationCondition: Send + Sync {
    /// Check if the group chat should end.
    async fn should_terminate(
        &self,
        history: &[Message],
        last_response: &AgentResponse,
        round: usize,
    ) -> bool;
}

/// Terminate after a fixed number of rounds.
pub struct MaxRoundsTermination {
    max_rounds: usize,
}

impl MaxRoundsTermination {
    pub fn new(max_rounds: usize) -> Self {
        Self { max_rounds }
    }
}

#[async_trait]
impl TerminationCondition for MaxRoundsTermination {
    async fn should_terminate(
        &self,
        _history: &[Message],
        _last_response: &AgentResponse,
        round: usize,
    ) -> bool {
        round >= self.max_rounds
    }
}

/// Terminate when the response contains a specific keyword (e.g., "TERMINATE").
pub struct KeywordTermination {
    keyword: String,
}

impl KeywordTermination {
    pub fn new(keyword: impl Into<String>) -> Self {
        Self {
            keyword: keyword.into(),
        }
    }
}

#[async_trait]
impl TerminationCondition for KeywordTermination {
    async fn should_terminate(
        &self,
        _history: &[Message],
        last_response: &AgentResponse,
        _round: usize,
    ) -> bool {
        last_response.text.contains(&self.keyword)
    }
}

/// Multi-agent group chat orchestration.
///
/// Agents take turns speaking in a shared conversation. A [`SpeakerSelector`]
/// determines who speaks next, and a [`TerminationCondition`] determines when
/// to stop.
///
/// Mirrors .NET's `GroupChatWorkflowBuilder` and Python's `GroupChatBuilder`.
pub struct GroupChatOrchestration {
    participants: Vec<GroupChatParticipant>,
    selector: Box<dyn SpeakerSelector>,
    termination: Box<dyn TerminationCondition>,
}

impl GroupChatOrchestration {
    pub fn new(
        selector: impl SpeakerSelector + 'static,
        termination: impl TerminationCondition + 'static,
    ) -> Self {
        Self {
            participants: Vec::new(),
            selector: Box::new(selector),
            termination: Box::new(termination),
        }
    }

    pub fn add_participant(mut self, participant: GroupChatParticipant) -> Self {
        self.participants.push(participant);
        self
    }

    /// Run the group chat.
    pub async fn run(
        &self,
        initial_message: Message,
        session: &mut AgentSession,
    ) -> AgentResult<Vec<AgentResponse>> {
        if self.participants.is_empty() {
            return Err(AgentError::InvalidRequest(
                "group chat has no participants".to_string(),
            ));
        }

        let mut history = vec![initial_message];
        let mut responses = Vec::new();

        for round in 0..100 {
            // Safety cap
            let speaker_idx = self
                .selector
                .select_next(&history, &self.participants, round)
                .await?;

            let participant = &self.participants[speaker_idx];

            // Build input: system message identifying the speaker + conversation history.
            let mut messages = vec![Message::system(format!(
                "You are '{}'. Respond as this participant in the group conversation.",
                participant.display_name
            ))];
            messages.extend(history.clone());

            let response = participant.agent.run(messages, session, None).await?;

            // Add the response to history with the speaker's name.
            let mut response_msg = Message::assistant(&response.text);
            response_msg.name = Some(participant.display_name.clone());
            history.push(response_msg);

            let should_stop = self
                .termination
                .should_terminate(&history, &response, round)
                .await;
            responses.push(response);

            if should_stop {
                break;
            }
        }

        Ok(responses)
    }
}
