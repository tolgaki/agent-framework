// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework Redis
//!
//! Redis persistence for the Microsoft Agent Framework.
//!
//! Provides [`RedisHistoryProvider`], a [`HistoryProvider`](agent_framework_core::session::HistoryProvider)
//! that stores conversation history in Redis, and [`RedisContextProvider`], a
//! [`ContextProvider`](agent_framework_core::context::ContextProvider) that
//! reads context data from Redis keys.
//!
//! # Example
//!
//! ```rust,no_run
//! use agent_framework_core::session::AgentSession;
//! use agent_framework_redis::{RedisConfig, RedisHistoryProvider};
//!
//! # async fn example() -> agent_framework_core::error::AgentResult<()> {
//! let config = RedisConfig::from_env()?;
//! let provider = RedisHistoryProvider::new(config).await?;
//! let session = AgentSession::with_history_provider(Box::new(provider));
//! # Ok(())
//! # }
//! ```

use async_trait::async_trait;
use redis::AsyncCommands;
use tokio::sync::Mutex;
use tracing::debug;

use agent_framework_core::context::{AIContext, ContextProvider};
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::session::{AgentSession, HistoryProvider};
use agent_framework_core::types::Message;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for connecting to Redis.
#[derive(Clone, Debug)]
pub struct RedisConfig {
    /// Redis connection URL. Defaults to `redis://127.0.0.1:6379`.
    pub url: String,

    /// Key prefix for all Redis keys. Defaults to `"agent:"`.
    pub key_prefix: String,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1:6379".into(),
            key_prefix: "agent:".into(),
        }
    }
}

impl RedisConfig {
    /// Create a config from environment variables.
    ///
    /// Reads `REDIS_URL` (optional, defaults to `redis://127.0.0.1:6379`).
    pub fn from_env() -> AgentResult<Self> {
        let url = std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".into());
        Ok(Self {
            url,
            ..Default::default()
        })
    }
}

// ---------------------------------------------------------------------------
// Connection helper
// ---------------------------------------------------------------------------

async fn open_connection(url: &str) -> AgentResult<redis::aio::MultiplexedConnection> {
    let client = redis::Client::open(url).map_err(|e| {
        AgentError::InvalidRequest(format!("Invalid Redis URL: {e}"))
    })?;
    client
        .get_multiplexed_async_connection()
        .await
        .map_err(|e| AgentError::HttpError(format!("Redis connection failed: {e}")))
}

// ---------------------------------------------------------------------------
// RedisHistoryProvider
// ---------------------------------------------------------------------------

/// A [`HistoryProvider`] backed by Redis.
///
/// Stores conversation messages as a JSON-serialized array in a Redis string
/// key. Key format: `{prefix}history:{session_id}`.
pub struct RedisHistoryProvider {
    config: RedisConfig,
    conn: Mutex<redis::aio::MultiplexedConnection>,
}

impl std::fmt::Debug for RedisHistoryProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisHistoryProvider")
            .field("config", &self.config)
            .finish()
    }
}

impl RedisHistoryProvider {
    /// Create a new provider, establishing a multiplexed Redis connection.
    pub async fn new(config: RedisConfig) -> AgentResult<Self> {
        let conn = open_connection(&config.url).await?;
        Ok(Self {
            config,
            conn: Mutex::new(conn),
        })
    }

    /// Build the Redis key for a session's history.
    fn history_key(&self, session_id: &str) -> String {
        format!("{}history:{}", self.config.key_prefix, session_id)
    }
}

#[async_trait]
impl HistoryProvider for RedisHistoryProvider {
    async fn get_history(&self, session_id: &str) -> AgentResult<Vec<Message>> {
        let key = self.history_key(session_id);
        debug!(session_id, key = %key, "Redis: fetching history");

        let mut conn = self.conn.lock().await;
        let value: Option<String> = conn.get(&key).await.map_err(|e| {
            AgentError::HttpError(format!("Redis GET failed: {e}"))
        })?;

        match value {
            Some(json) => {
                let messages: Vec<Message> = serde_json::from_str(&json).map_err(|e| {
                    AgentError::InvalidResponse(format!(
                        "Failed to parse history from Redis: {e}"
                    ))
                })?;
                Ok(messages)
            }
            None => Ok(Vec::new()),
        }
    }

    async fn save_history(&self, session_id: &str, messages: &[Message]) -> AgentResult<()> {
        let key = self.history_key(session_id);

        // Read existing history, append new messages, write back.
        let mut existing = self.get_history(session_id).await.unwrap_or_default();
        existing.extend(messages.iter().cloned());

        let json = serde_json::to_string(&existing)?;

        debug!(
            session_id,
            key = %key,
            count = existing.len(),
            "Redis: saving history"
        );

        let mut conn = self.conn.lock().await;
        conn.set::<_, _, ()>(&key, &json).await.map_err(|e| {
            AgentError::HttpError(format!("Redis SET failed: {e}"))
        })?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RedisContextProvider
// ---------------------------------------------------------------------------

/// A [`ContextProvider`] that reads context from a Redis key and injects
/// it as system instructions.
///
/// The context is stored as a plain string in Redis at the key
/// `{prefix}context:{session_id}`. If the key is absent, no extra context
/// is injected.
pub struct RedisContextProvider {
    config: RedisConfig,
    conn: Mutex<redis::aio::MultiplexedConnection>,
}

impl RedisContextProvider {
    /// Create a new context provider, establishing a multiplexed Redis connection.
    pub async fn new(config: RedisConfig) -> AgentResult<Self> {
        let conn = open_connection(&config.url).await?;
        Ok(Self {
            config,
            conn: Mutex::new(conn),
        })
    }

    /// Build the Redis key for a session's context.
    fn context_key(&self, session_id: &str) -> String {
        format!("{}context:{}", self.config.key_prefix, session_id)
    }
}

#[async_trait]
impl ContextProvider for RedisContextProvider {
    async fn provide_context(&self, session: &AgentSession) -> AgentResult<AIContext> {
        let key = self.context_key(&session.session_id);
        debug!(session_id = %session.session_id, key = %key, "Redis: fetching context");

        let mut conn = self.conn.lock().await;
        let value: Option<String> = conn.get(&key).await.map_err(|e| {
            AgentError::HttpError(format!("Redis GET (context) failed: {e}"))
        })?;

        match value {
            Some(instructions) if !instructions.is_empty() => {
                debug!(
                    session_id = %session.session_id,
                    len = instructions.len(),
                    "Redis: injecting context instructions"
                );
                Ok(AIContext {
                    instructions: Some(instructions),
                    ..Default::default()
                })
            }
            _ => Ok(AIContext::default()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default() {
        let config = RedisConfig::default();
        assert_eq!(config.url, "redis://127.0.0.1:6379");
        assert_eq!(config.key_prefix, "agent:");
    }

    #[test]
    fn config_from_env_uses_default_when_unset() {
        std::env::remove_var("REDIS_URL");
        let config = RedisConfig::from_env().unwrap();
        assert_eq!(config.url, "redis://127.0.0.1:6379");
        assert_eq!(config.key_prefix, "agent:");
    }

    #[test]
    fn config_from_env_reads_url() {
        std::env::set_var("REDIS_URL", "redis://custom:1234");
        let config = RedisConfig::from_env().unwrap();
        assert_eq!(config.url, "redis://custom:1234");

        // Clean up.
        std::env::remove_var("REDIS_URL");
    }

    #[test]
    fn history_key_format() {
        let config = RedisConfig {
            url: "redis://localhost".into(),
            key_prefix: "myapp:".into(),
        };
        // We test the key format without needing a real connection.
        let key = format!("{}history:{}", config.key_prefix, "session-42");
        assert_eq!(key, "myapp:history:session-42");
    }

    #[test]
    fn context_key_format() {
        let config = RedisConfig::default();
        let key = format!("{}context:{}", config.key_prefix, "s1");
        assert_eq!(key, "agent:context:s1");
    }

    #[test]
    fn config_debug_display() {
        let config = RedisConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("redis://127.0.0.1:6379"));
        assert!(debug.contains("agent:"));
    }
}
