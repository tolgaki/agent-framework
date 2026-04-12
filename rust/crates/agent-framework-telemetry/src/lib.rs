// Copyright (c) Microsoft. All rights reserved.

//! OpenTelemetry-style instrumentation for the Agent Framework.
//!
//! This crate provides tracing-based observability components that mirror .NET's
//! `OpenTelemetryAgent` and Python's observability module. It uses the [`tracing`]
//! crate as the instrumentation layer — spans emitted here bridge to OpenTelemetry
//! collectors via [`tracing_opentelemetry`](https://docs.rs/tracing-opentelemetry).
//!
//! # Components
//!
//! - [`TelemetryAgent`] — Wraps any [`Agent`] to record spans and metrics for
//!   every `run()` call (agent-level instrumentation).
//! - [`TelemetryChatMiddleware`] — A [`ChatClientMiddleware`] that instruments
//!   individual model calls.
//! - [`TelemetryFunctionMiddleware`] — A [`FunctionMiddleware`] that instruments
//!   tool invocations.
//!
//! # Semantic conventions
//!
//! Field names follow the [OpenTelemetry GenAI semantic conventions][otel-genai]
//! and are exposed as constants in the [`conventions`] module.
//!
//! [otel-genai]: https://opentelemetry.io/docs/specs/semconv/gen-ai/

pub mod conventions;

use std::time::Instant;

use async_trait::async_trait;
use tracing::{info_span, Instrument};

use agent_framework_core::agent::Agent;
use agent_framework_core::client::ChatClient;
use agent_framework_core::error::AgentResult;
use agent_framework_core::middleware::{ChatClientMiddleware, FunctionMiddleware, FunctionMiddlewareNext};
use agent_framework_core::session::AgentSession;
use agent_framework_core::streaming::AgentResponseStream;
use agent_framework_core::tools::FunctionTool;
use agent_framework_core::types::{
    AgentResponse, AgentRunOptions, ChatOptions, ChatResponse, Message,
};

use conventions::*;

// ---------------------------------------------------------------------------
// TelemetryAgent
// ---------------------------------------------------------------------------

/// A delegating agent wrapper that records tracing spans and metrics for every run.
///
/// Creates a span with agent identity and input details on entry, then records
/// output details (message counts, token usage, finish reason, duration) on exit.
/// Errors are captured as span events.
///
/// This mirrors .NET's `OpenTelemetryAgent` and Python's observability wrappers.
///
/// # Example
///
/// ```rust,no_run
/// use agent_framework_telemetry::TelemetryAgent;
/// # use agent_framework_core::agent::Agent;
///
/// # fn example(inner: Box<dyn Agent>) {
/// let agent = TelemetryAgent::new(inner);
/// // `agent` implements `Agent` and emits tracing spans on every run.
/// # }
/// ```
pub struct TelemetryAgent {
    inner: Box<dyn Agent>,
}

impl TelemetryAgent {
    /// Wrap an existing agent with telemetry instrumentation.
    pub fn new(inner: Box<dyn Agent>) -> Self {
        Self { inner }
    }

    /// Wrap an agent value (moves it into a `Box`).
    pub fn wrap(inner: impl Agent + 'static) -> Self {
        Self {
            inner: Box::new(inner),
        }
    }
}

#[async_trait]
impl Agent for TelemetryAgent {
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
        let agent_id = self.inner.id().to_owned();
        let agent_name = self.inner.name().unwrap_or("").to_owned();
        let input_message_count = messages.len() as u64;

        let span = info_span!(
            "agent.run",
            { AGENT_ID } = %agent_id,
            { AGENT_NAME } = %agent_name,
            { GEN_AI_OPERATION_NAME } = "agent.run",
            input_message_count = input_message_count,
            output_message_count = tracing::field::Empty,
            finish_reason = tracing::field::Empty,
            { GEN_AI_USAGE_INPUT_TOKENS } = tracing::field::Empty,
            { GEN_AI_USAGE_OUTPUT_TOKENS } = tracing::field::Empty,
            { GEN_AI_OPERATION_DURATION } = tracing::field::Empty,
        );

        let start = Instant::now();

        let result = self
            .inner
            .run(messages, session, options)
            .instrument(span.clone())
            .await;

        let duration_ms = start.elapsed().as_millis() as u64;
        span.record(GEN_AI_OPERATION_DURATION, duration_ms);

        match &result {
            Ok(response) => {
                span.record("output_message_count", response.messages.len() as u64);
                span.record(
                    "finish_reason",
                    format!("{:?}", response.finish_reason).as_str(),
                );
                if let Some(usage) = &response.usage {
                    span.record(GEN_AI_USAGE_INPUT_TOKENS, u64::from(usage.input_tokens));
                    span.record(GEN_AI_USAGE_OUTPUT_TOKENS, u64::from(usage.output_tokens));
                }
            }
            Err(err) => {
                let _entered = span.enter();
                tracing::error!(
                    error = %err,
                    "Agent run failed"
                );
            }
        }

        result
    }

    fn run_stream<'a>(
        &'a self,
        messages: Vec<Message>,
        session: &'a mut AgentSession,
        options: Option<&'a AgentRunOptions>,
    ) -> AgentResult<AgentResponseStream<'a>> {
        // Streaming delegates directly; agent-level hooks are not applied
        // during streaming (consistent with DelegatingAgent and AgentMiddleware).
        self.inner.run_stream(messages, session, options)
    }
}

// ---------------------------------------------------------------------------
// TelemetryChatMiddleware
// ---------------------------------------------------------------------------

/// A [`ChatClientMiddleware`] that instruments model calls with tracing spans.
///
/// Records the model name, token usage, and call duration for each
/// `get_response` invocation. Mirrors the chat-level telemetry in .NET's
/// `OpenTelemetryChatClient` and Python's chat observability layer.
pub struct TelemetryChatMiddleware;

impl TelemetryChatMiddleware {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TelemetryChatMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ChatClientMiddleware for TelemetryChatMiddleware {
    async fn on_get_response(
        &self,
        messages: &mut Vec<Message>,
        options: &mut ChatOptions,
        next: &dyn ChatClient,
    ) -> AgentResult<ChatResponse> {
        let model = options
            .model
            .as_deref()
            .unwrap_or("unknown")
            .to_owned();

        let span = info_span!(
            "chat.completion",
            { GEN_AI_SYSTEM } = "agent_framework",
            { GEN_AI_OPERATION_NAME } = "chat.completion",
            { GEN_AI_REQUEST_MODEL } = %model,
            { GEN_AI_RESPONSE_MODEL } = tracing::field::Empty,
            { GEN_AI_USAGE_INPUT_TOKENS } = tracing::field::Empty,
            { GEN_AI_USAGE_OUTPUT_TOKENS } = tracing::field::Empty,
            { GEN_AI_OPERATION_DURATION } = tracing::field::Empty,
        );

        let start = Instant::now();

        let result = next
            .get_response(messages, Some(options))
            .instrument(span.clone())
            .await;

        let duration_ms = start.elapsed().as_millis() as u64;
        span.record(GEN_AI_OPERATION_DURATION, duration_ms);

        match &result {
            Ok(response) => {
                // The response model may differ from the request model
                // (e.g. auto-routing). Record it if available.
                span.record(GEN_AI_RESPONSE_MODEL, model.as_str());

                if let Some(usage) = &response.usage {
                    span.record(GEN_AI_USAGE_INPUT_TOKENS, u64::from(usage.input_tokens));
                    span.record(GEN_AI_USAGE_OUTPUT_TOKENS, u64::from(usage.output_tokens));
                }
            }
            Err(err) => {
                let _entered = span.enter();
                tracing::error!(
                    error = %err,
                    "Chat completion failed"
                );
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// TelemetryFunctionMiddleware
// ---------------------------------------------------------------------------

/// A [`FunctionMiddleware`] that instruments tool invocations with tracing spans.
///
/// Records the tool name, invocation duration, and success/failure status.
/// Mirrors Python's function-level observability and .NET's tool telemetry.
pub struct TelemetryFunctionMiddleware;

impl TelemetryFunctionMiddleware {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TelemetryFunctionMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FunctionMiddleware for TelemetryFunctionMiddleware {
    async fn on_invoke(
        &self,
        tool: &dyn FunctionTool,
        args: serde_json::Value,
        next: &dyn FunctionMiddlewareNext,
    ) -> AgentResult<serde_json::Value> {
        let tool_name = tool.definition().name.clone();

        let span = info_span!(
            "tool.invoke",
            { GEN_AI_OPERATION_NAME } = "tool.invoke",
            tool.name = %tool_name,
            { GEN_AI_OPERATION_DURATION } = tracing::field::Empty,
            tool.success = tracing::field::Empty,
        );

        let start = Instant::now();

        let result = {
            let _entered = span.enter();
            next.invoke(args).await
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        span.record(GEN_AI_OPERATION_DURATION, duration_ms);
        span.record("tool.success", result.is_ok());

        if let Err(err) = &result {
            let _entered = span.enter();
            tracing::error!(
                error = %err,
                tool.name = %tool_name,
                "Tool invocation failed"
            );
        }

        result
    }
}
