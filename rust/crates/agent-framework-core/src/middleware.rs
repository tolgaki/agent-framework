// Copyright (c) Microsoft. All rights reserved.

//! Three-layer middleware pipeline.
//!
//! The pipeline lets callers wrap agent execution with cross-cutting concerns:
//!
//! - [`AgentMiddleware`] wraps the entire agent run (outermost layer).
//! - [`ChatClientMiddleware`] wraps individual chat-client calls.
//! - [`FunctionMiddleware`] wraps tool invocations.
//!
//! Agent and function middleware are invoked via a recursive `next` handler.
//! Chat-client middleware uses the decorator pattern: each middleware becomes
//! a [`ChatClient`] that wraps an inner client, chaining them at build time.

use async_trait::async_trait;

use crate::client::ChatClient;
use crate::error::AgentResult;
use crate::session::AgentSession;
use crate::streaming::ResponseStream;
use crate::tools::FunctionTool;
use crate::types::{AgentResponse, ChatOptions, ChatResponse, Message};

// ---------------------------------------------------------------------------
// Agent-level middleware
// ---------------------------------------------------------------------------

/// Agent-level middleware that wraps the entire agent run.
///
/// This is the outermost layer, corresponding to Python's `AgentMiddlewareLayer`
/// and .NET's `DelegatingAIAgent` decorator pattern.
#[async_trait]
pub trait AgentMiddleware: Send + Sync {
    /// Process an agent run, optionally delegating to `next`.
    async fn on_run(
        &self,
        messages: Vec<Message>,
        session: &mut AgentSession,
        next: &dyn AgentMiddlewareNext,
    ) -> AgentResult<AgentResponse>;
}

/// The next handler in the agent middleware chain.
#[async_trait]
pub trait AgentMiddlewareNext: Send + Sync {
    /// Continue to the next middleware or the inner agent.
    async fn run(&self, messages: Vec<Message>, session: &mut AgentSession) -> AgentResult<AgentResponse>;
}

/// Internal chain node for the agent middleware pipeline.
///
/// Walks `remaining` one entry at a time; when the slice is empty, defers to
/// `terminal` (which invokes the wrapped agent's inner run).
pub(crate) struct AgentChain<'a> {
    pub(crate) remaining: &'a [Box<dyn AgentMiddleware>],
    pub(crate) terminal: &'a dyn AgentMiddlewareNext,
}

#[async_trait]
impl<'a> AgentMiddlewareNext for AgentChain<'a> {
    async fn run(&self, messages: Vec<Message>, session: &mut AgentSession) -> AgentResult<AgentResponse> {
        if let Some((first, rest)) = self.remaining.split_first() {
            let next = AgentChain {
                remaining: rest,
                terminal: self.terminal,
            };
            first.on_run(messages, session, &next).await
        } else {
            self.terminal.run(messages, session).await
        }
    }
}

// ---------------------------------------------------------------------------
// Chat-client-level middleware (decorator pattern)
// ---------------------------------------------------------------------------

/// Chat-client-level middleware that wraps individual model calls.
///
/// Corresponds to Python's `ChatMiddlewareLayer` and .NET's
/// `DelegatingChatClient` pattern.
#[async_trait]
pub trait ChatClientMiddleware: Send + Sync {
    /// Process a chat completion request. Call `next.get_response(...)` to proceed.
    async fn on_get_response(
        &self,
        messages: &mut Vec<Message>,
        options: &mut ChatOptions,
        next: &dyn ChatClient,
    ) -> AgentResult<ChatResponse>;

    /// Process a streaming chat completion request.
    ///
    /// The default implementation forwards directly to `next` without modification.
    /// Override to intercept streaming calls.
    async fn on_get_response_stream(
        &self,
        messages: &mut Vec<Message>,
        options: &mut ChatOptions,
        next: &dyn ChatClient,
    ) -> AgentResult<ResponseStream> {
        next.get_response_stream(messages, Some(options)).await
    }
}

/// A [`ChatClient`] decorator that applies a single [`ChatClientMiddleware`]
/// around calls to an inner client.
///
/// Used internally to build a chain of chat-client middleware at agent
/// construction time.
pub struct ChatClientDecorator {
    middleware: Box<dyn ChatClientMiddleware>,
    inner: Box<dyn ChatClient>,
}

impl ChatClientDecorator {
    /// Wrap `inner` with `middleware`.
    pub fn new(middleware: Box<dyn ChatClientMiddleware>, inner: Box<dyn ChatClient>) -> Self {
        Self { middleware, inner }
    }
}

#[async_trait]
impl ChatClient for ChatClientDecorator {
    async fn get_response(&self, messages: &[Message], options: Option<&ChatOptions>) -> AgentResult<ChatResponse> {
        let mut msgs = messages.to_vec();
        let mut opts = options.cloned().unwrap_or_default();
        self.middleware
            .on_get_response(&mut msgs, &mut opts, self.inner.as_ref())
            .await
    }

    async fn get_response_stream(
        &self,
        messages: &[Message],
        options: Option<&ChatOptions>,
    ) -> AgentResult<ResponseStream> {
        let mut msgs = messages.to_vec();
        let mut opts = options.cloned().unwrap_or_default();
        self.middleware
            .on_get_response_stream(&mut msgs, &mut opts, self.inner.as_ref())
            .await
    }
}

// ---------------------------------------------------------------------------
// Function-level middleware
// ---------------------------------------------------------------------------

/// Function-level middleware that wraps tool invocations.
///
/// Corresponds to Python's `FunctionMiddlewareLayer`. Useful for approval gates,
/// logging, or modifying tool inputs/outputs.
#[async_trait]
pub trait FunctionMiddleware: Send + Sync {
    /// Process a tool invocation. Call `next.invoke(args)` to proceed.
    async fn on_invoke(
        &self,
        tool: &dyn FunctionTool,
        args: serde_json::Value,
        next: &dyn FunctionMiddlewareNext,
    ) -> AgentResult<serde_json::Value>;
}

/// The next handler in the function middleware chain.
#[async_trait]
pub trait FunctionMiddlewareNext: Send + Sync {
    /// Continue to the next middleware or the actual tool.
    async fn invoke(&self, args: serde_json::Value) -> AgentResult<serde_json::Value>;
}

/// Internal chain node for the function middleware pipeline.
pub(crate) struct FunctionChain<'a> {
    pub(crate) remaining: &'a [Box<dyn FunctionMiddleware>],
    pub(crate) tool: &'a dyn FunctionTool,
}

#[async_trait]
impl<'a> FunctionMiddlewareNext for FunctionChain<'a> {
    async fn invoke(&self, args: serde_json::Value) -> AgentResult<serde_json::Value> {
        if let Some((first, rest)) = self.remaining.split_first() {
            let next = FunctionChain {
                remaining: rest,
                tool: self.tool,
            };
            first.on_invoke(self.tool, args, &next).await
        } else {
            self.tool.invoke(args).await
        }
    }
}

// ---------------------------------------------------------------------------
// Pipeline container
// ---------------------------------------------------------------------------

/// An ordered pipeline of middleware layers.
///
/// Middleware is executed in the order it was added. The first layer added
/// is the outermost (called first, returns last).
#[derive(Default)]
pub struct MiddlewarePipeline {
    /// Agent-level middleware, executed outermost-first.
    pub agent_middleware: Vec<Box<dyn AgentMiddleware>>,

    /// Chat-client-level middleware, applied as decorators around the client.
    pub chat_client_middleware: Vec<Box<dyn ChatClientMiddleware>>,

    /// Function-level middleware, executed around each tool call.
    pub function_middleware: Vec<Box<dyn FunctionMiddleware>>,
}

impl MiddlewarePipeline {
    /// Create an empty middleware pipeline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add agent-level middleware.
    pub fn add_agent_middleware(&mut self, middleware: impl AgentMiddleware + 'static) {
        self.agent_middleware.push(Box::new(middleware));
    }

    /// Add chat-client-level middleware.
    pub fn add_chat_client_middleware(&mut self, middleware: impl ChatClientMiddleware + 'static) {
        self.chat_client_middleware.push(Box::new(middleware));
    }

    /// Add function-level middleware.
    pub fn add_function_middleware(&mut self, middleware: impl FunctionMiddleware + 'static) {
        self.function_middleware.push(Box::new(middleware));
    }
}
