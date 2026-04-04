// Copyright (c) Microsoft. All rights reserved.

use async_trait::async_trait;

use crate::client::ChatClient;
use crate::error::AgentResult;
use crate::session::AgentSession;
use crate::tools::FunctionTool;
use crate::types::{AgentResponse, ChatOptions, ChatResponse, Message};

/// Agent-level middleware that wraps the entire agent run.
///
/// This is the outermost layer, corresponding to Python's `AgentMiddlewareLayer`
/// and .NET's decorator agents (e.g., `LoggingAgent`, `OpenTelemetryAgent`).
///
/// Middleware can inspect/modify messages, short-circuit the run, or add
/// post-processing after the inner agent completes.
#[async_trait]
pub trait AgentMiddleware: Send + Sync {
    /// Process an agent run, optionally delegating to `next`.
    ///
    /// Call `next.run(messages, session).await` to proceed to the next layer.
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

/// Chat-client-level middleware that wraps individual model calls.
///
/// Corresponds to Python's `ChatMiddlewareLayer` and .NET's
/// `DelegatingChatClient` pattern.
#[async_trait]
pub trait ChatClientMiddleware: Send + Sync {
    /// Process a chat completion request.
    async fn on_get_response(
        &self,
        messages: &mut Vec<Message>,
        options: &mut ChatOptions,
        next: &dyn ChatClient,
    ) -> AgentResult<ChatResponse>;
}

/// Function-level middleware that wraps tool invocations.
///
/// Corresponds to Python's `FunctionMiddlewareLayer`. Useful for approval gates,
/// logging, or modifying tool inputs/outputs.
#[async_trait]
pub trait FunctionMiddleware: Send + Sync {
    /// Process a tool invocation.
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

/// An ordered pipeline of middleware layers.
///
/// Middleware is executed in the order it was added. Each layer can choose
/// to call `next` to continue the chain or return early.
pub struct MiddlewarePipeline {
    /// Agent-level middleware, executed outermost-first.
    pub agent_middleware: Vec<Box<dyn AgentMiddleware>>,

    /// Chat-client-level middleware.
    pub chat_client_middleware: Vec<Box<dyn ChatClientMiddleware>>,

    /// Function-level middleware, executed around each tool call.
    pub function_middleware: Vec<Box<dyn FunctionMiddleware>>,
}

impl MiddlewarePipeline {
    /// Create an empty middleware pipeline.
    pub fn new() -> Self {
        Self {
            agent_middleware: Vec::new(),
            chat_client_middleware: Vec::new(),
            function_middleware: Vec::new(),
        }
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

impl Default for MiddlewarePipeline {
    fn default() -> Self {
        Self::new()
    }
}
