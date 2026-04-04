// Copyright (c) Microsoft. All rights reserved.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::error::AgentResult;
use crate::types::Message;

/// A conversation session that tracks state across agent runs.
///
/// Corresponds to Python's `AgentSession` and .NET's `AgentSession`.
#[derive(Debug)]
pub struct AgentSession {
    /// Unique identifier for this session.
    pub session_id: String,

    /// Arbitrary key-value state storage.
    pub state: HashMap<String, serde_json::Value>,

    /// The history provider used to persist conversation messages.
    history_provider: Box<dyn HistoryProvider>,
}

impl AgentSession {
    /// Create a new session with an in-memory history provider.
    pub fn new() -> Self {
        Self {
            session_id: uuid::Uuid::new_v4().to_string(),
            state: HashMap::new(),
            history_provider: Box::new(InMemoryHistoryProvider::new()),
        }
    }

    /// Create a new session with a specific history provider.
    pub fn with_history_provider(history_provider: Box<dyn HistoryProvider>) -> Self {
        Self {
            session_id: uuid::Uuid::new_v4().to_string(),
            state: HashMap::new(),
            history_provider,
        }
    }

    /// Load the conversation history for this session.
    pub async fn get_history(&self) -> AgentResult<Vec<Message>> {
        self.history_provider.get_history(&self.session_id).await
    }

    /// Save messages to the conversation history.
    pub async fn save_history(&self, messages: &[Message]) -> AgentResult<()> {
        self.history_provider.save_history(&self.session_id, messages).await
    }
}

impl Default for AgentSession {
    fn default() -> Self {
        Self::new()
    }
}

/// A provider for loading and persisting conversation history.
///
/// Corresponds to Python's `HistoryProvider` protocol and .NET's `ChatHistoryProvider`.
#[async_trait]
pub trait HistoryProvider: Send + Sync + std::fmt::Debug {
    /// Load conversation history for the given session.
    async fn get_history(&self, session_id: &str) -> AgentResult<Vec<Message>>;

    /// Append messages to the conversation history.
    async fn save_history(&self, session_id: &str, messages: &[Message]) -> AgentResult<()>;
}

/// An in-memory history provider for development and testing.
///
/// Stores messages in a thread-safe map. Data is lost when the process exits.
#[derive(Debug, Clone)]
pub struct InMemoryHistoryProvider {
    store: Arc<RwLock<HashMap<String, Vec<Message>>>>,
}

impl InMemoryHistoryProvider {
    /// Create a new empty in-memory history store.
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryHistoryProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl HistoryProvider for InMemoryHistoryProvider {
    async fn get_history(&self, session_id: &str) -> AgentResult<Vec<Message>> {
        let store = self.store.read().await;
        Ok(store.get(session_id).cloned().unwrap_or_default())
    }

    async fn save_history(&self, session_id: &str, messages: &[Message]) -> AgentResult<()> {
        let mut store = self.store.write().await;
        store
            .entry(session_id.to_string())
            .or_default()
            .extend(messages.iter().cloned());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_history() {
        let provider = InMemoryHistoryProvider::new();
        let session_id = "test-session";

        // Initially empty.
        let history = provider.get_history(session_id).await.unwrap();
        assert!(history.is_empty());

        // Save some messages.
        let messages = vec![Message::user("Hello"), Message::assistant("Hi there!")];
        provider.save_history(session_id, &messages).await.unwrap();

        // Retrieve them.
        let history = provider.get_history(session_id).await.unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].text(), "Hello");
        assert_eq!(history[1].text(), "Hi there!");

        // Append more.
        provider
            .save_history(session_id, &[Message::user("How are you?")])
            .await
            .unwrap();
        let history = provider.get_history(session_id).await.unwrap();
        assert_eq!(history.len(), 3);
    }

    #[tokio::test]
    async fn test_session_default() {
        let session = AgentSession::new();
        assert!(!session.session_id.is_empty());

        let history = session.get_history().await.unwrap();
        assert!(history.is_empty());
    }
}
