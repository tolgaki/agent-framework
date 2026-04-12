// Copyright (c) Microsoft. All rights reserved.

//! Delegating agent wrappers for cross-cutting concerns.
//!
//! A [`DelegatingAgent`] wraps an inner agent and adds behaviour around each
//! run — logging, metrics, retry logic, etc.  This mirrors .NET's
//! `DelegatingAIAgent` pattern and complements the middleware system with an
//! agent-level decorator that is simpler to compose for one-off concerns.
//!
//! # Built-in delegates
//!
//! - [`LoggingAgent`] — logs entry/exit and timing for every run.

use std::time::Instant;

use async_trait::async_trait;
use tracing::{debug, info, warn};

use crate::agent::Agent;
use crate::error::AgentResult;
use crate::session::AgentSession;
use crate::streaming::AgentResponseStream;
use crate::types::{AgentResponse, AgentRunOptions, Message};

/// An agent decorator that delegates to an inner agent, adding cross-cutting
/// behaviour.
///
/// Implement this trait to add logging, metrics, retries, guardrails, etc.
///
/// Corresponds to .NET's `DelegatingAIAgent`.
#[async_trait]
pub trait DelegatingAgent: Send + Sync {
    /// The inner agent this delegate wraps.
    fn inner(&self) -> &dyn Agent;

    /// Pre-processing hook called before the inner agent runs.
    /// Return `Some(response)` to short-circuit (skip the inner agent).
    async fn before_run(
        &self,
        _messages: &[Message],
        _session: &AgentSession,
        _options: Option<&AgentRunOptions>,
    ) -> AgentResult<Option<AgentResponse>> {
        Ok(None)
    }

    /// Post-processing hook called after the inner agent runs.
    async fn after_run(
        &self,
        _response: &AgentResponse,
        _session: &AgentSession,
    ) -> AgentResult<()> {
        Ok(())
    }
}

// Blanket Agent impl for any DelegatingAgent.
#[async_trait]
impl<T: DelegatingAgent> Agent for T {
    fn id(&self) -> &str {
        self.inner().id()
    }

    fn name(&self) -> Option<&str> {
        self.inner().name()
    }

    fn description(&self) -> Option<&str> {
        self.inner().description()
    }

    async fn run(
        &self,
        messages: Vec<Message>,
        session: &mut AgentSession,
        options: Option<&AgentRunOptions>,
    ) -> AgentResult<AgentResponse> {
        // Pre-hook: allow short-circuit.
        if let Some(response) = self.before_run(&messages, session, options).await? {
            return Ok(response);
        }

        let response = self.inner().run(messages, session, options).await?;

        // Post-hook.
        self.after_run(&response, session).await?;

        Ok(response)
    }

    fn run_stream<'a>(
        &'a self,
        messages: Vec<Message>,
        session: &'a mut AgentSession,
        options: Option<&'a AgentRunOptions>,
    ) -> AgentResult<AgentResponseStream<'a>> {
        // Streaming delegates to the inner agent — hooks are not applied
        // during streaming (same limitation as agent middleware).
        self.inner().run_stream(messages, session, options)
    }
}

// ---------------------------------------------------------------------------
// LoggingAgent
// ---------------------------------------------------------------------------

/// A delegating agent that logs entry, exit, and timing for every run.
///
/// Corresponds to .NET's `LoggingAgent`.
pub struct LoggingAgent {
    inner: Box<dyn Agent>,
}

impl LoggingAgent {
    pub fn new(inner: impl Agent + 'static) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }

    pub fn wrap(inner: Box<dyn Agent>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl DelegatingAgent for LoggingAgent {
    fn inner(&self) -> &dyn Agent {
        self.inner.as_ref()
    }

    async fn before_run(
        &self,
        messages: &[Message],
        _session: &AgentSession,
        _options: Option<&AgentRunOptions>,
    ) -> AgentResult<Option<AgentResponse>> {
        info!(
            agent_id = %self.inner.id(),
            agent_name = ?self.inner.name(),
            input_messages = messages.len(),
            "Agent run starting"
        );
        Ok(None)
    }

    async fn after_run(
        &self,
        response: &AgentResponse,
        _session: &AgentSession,
    ) -> AgentResult<()> {
        let usage = response.usage.as_ref();
        info!(
            agent_id = %self.inner.id(),
            output_messages = response.messages.len(),
            finish_reason = ?response.finish_reason,
            input_tokens = usage.map(|u| u.input_tokens).unwrap_or(0),
            output_tokens = usage.map(|u| u.output_tokens).unwrap_or(0),
            "Agent run completed"
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RetryAgent
// ---------------------------------------------------------------------------

/// A delegating agent that retries on transient errors.
pub struct RetryAgent {
    inner: Box<dyn Agent>,
    max_retries: usize,
}

impl RetryAgent {
    pub fn new(inner: impl Agent + 'static, max_retries: usize) -> Self {
        Self {
            inner: Box::new(inner),
            max_retries,
        }
    }
}

#[async_trait]
impl Agent for RetryAgent {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn name(&self) -> Option<&str> {
        self.inner.name()
    }

    fn description(&self) -> Option<&str> {
        self.inner.description()
    }

    async fn run(
        &self,
        messages: Vec<Message>,
        session: &mut AgentSession,
        options: Option<&AgentRunOptions>,
    ) -> AgentResult<AgentResponse> {
        let mut last_err = None;
        for attempt in 0..=self.max_retries {
            match self.inner.run(messages.clone(), session, options).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if attempt < self.max_retries && is_retryable(&e) {
                        let delay = std::time::Duration::from_millis(100 * 2u64.pow(attempt as u32));
                        warn!(
                            attempt = attempt + 1,
                            max_retries = self.max_retries,
                            delay_ms = delay.as_millis() as u64,
                            error = %e,
                            "Retrying agent run"
                        );
                        tokio::time::sleep(delay).await;
                        last_err = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        Err(last_err.unwrap())
    }

    fn run_stream<'a>(
        &'a self,
        messages: Vec<Message>,
        session: &'a mut AgentSession,
        options: Option<&'a AgentRunOptions>,
    ) -> AgentResult<AgentResponseStream<'a>> {
        self.inner.run_stream(messages, session, options)
    }
}

fn is_retryable(err: &crate::error::AgentError) -> bool {
    match err {
        crate::error::AgentError::ProviderError { status_code, .. } => {
            matches!(status_code, Some(429) | Some(500) | Some(502) | Some(503) | Some(504))
        }
        crate::error::AgentError::HttpError(_) => true,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// TimingAgent
// ---------------------------------------------------------------------------

/// A delegating agent that measures and logs execution time.
pub struct TimingAgent {
    inner: Box<dyn Agent>,
}

impl TimingAgent {
    pub fn new(inner: impl Agent + 'static) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }
}

#[async_trait]
impl Agent for TimingAgent {
    fn id(&self) -> &str {
        self.inner.id()
    }

    fn name(&self) -> Option<&str> {
        self.inner.name()
    }

    fn description(&self) -> Option<&str> {
        self.inner.description()
    }

    async fn run(
        &self,
        messages: Vec<Message>,
        session: &mut AgentSession,
        options: Option<&AgentRunOptions>,
    ) -> AgentResult<AgentResponse> {
        let start = Instant::now();
        let result = self.inner.run(messages, session, options).await;
        let elapsed = start.elapsed();
        debug!(
            agent_id = %self.inner.id(),
            elapsed_ms = elapsed.as_millis() as u64,
            success = result.is_ok(),
            "Agent run timing"
        );
        result
    }

    fn run_stream<'a>(
        &'a self,
        messages: Vec<Message>,
        session: &'a mut AgentSession,
        options: Option<&'a AgentRunOptions>,
    ) -> AgentResult<AgentResponseStream<'a>> {
        self.inner.run_stream(messages, session, options)
    }
}
