// Copyright (c) Microsoft. All rights reserved.

use async_trait::async_trait;

use crate::error::AgentResult;
use crate::session::AgentSession;
use crate::tools::FunctionTool;
use crate::types::Message;

/// Additional context injected into an agent run.
///
/// Corresponds to Python's context providers and .NET's `AIContext`.
#[derive(Default)]
pub struct AIContext {
    /// Additional system instructions to prepend.
    pub instructions: Option<String>,

    /// Additional context messages to include.
    pub messages: Vec<Message>,

    /// Additional tools available for this run.
    pub tools: Vec<Box<dyn FunctionTool>>,
}

/// A provider that supplies additional context before an agent run.
///
/// Context providers are invoked before the model call, allowing dynamic
/// injection of instructions, messages (e.g., RAG results), and tools.
///
/// Corresponds to Python's `ContextProvider` and .NET's `AIContextProvider`.
#[async_trait]
pub trait ContextProvider: Send + Sync {
    /// Provide context for the current agent run.
    async fn provide_context(&self, session: &AgentSession) -> AgentResult<AIContext>;
}
