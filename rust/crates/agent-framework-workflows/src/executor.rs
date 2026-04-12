// Copyright (c) Microsoft. All rights reserved.

//! Workflow node executors.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;

use agent_framework_core::agent::Agent;
use agent_framework_core::error::AgentResult;
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::Message;

use crate::state::WorkflowState;

/// Type alias for async executor functions.
type ExecutorFn = Box<dyn Fn(&WorkflowState) -> Pin<Box<dyn Future<Output = AgentResult<()>> + Send + '_>> + Send + Sync>;

/// A node in the workflow DAG that performs work.
///
/// Mirrors .NET's workflow `Executor` and Python's `Executor` / `@handler`.
#[async_trait]
pub trait Executor: Send + Sync {
    /// The node's unique identifier within the workflow.
    fn id(&self) -> &str;

    /// Execute the node's logic, reading/writing shared [`WorkflowState`].
    async fn execute(&self, state: &WorkflowState) -> AgentResult<()>;
}

// ---------------------------------------------------------------------------
// FunctionExecutor
// ---------------------------------------------------------------------------

/// An executor backed by an async function.
///
/// Mirrors Python's `FunctionExecutor` / `@executor` decorator.
pub struct FunctionExecutor {
    id: String,
    func: ExecutorFn,
}

impl FunctionExecutor {
    /// Create a function executor with the given ID and async closure.
    pub fn new<F>(func: F) -> Self
    where
        F: Fn(&WorkflowState) -> Pin<Box<dyn Future<Output = AgentResult<()>> + Send + '_>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            func: Box::new(func),
        }
    }

    /// Set a custom ID for this executor.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }
}

#[async_trait]
impl Executor for FunctionExecutor {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, state: &WorkflowState) -> AgentResult<()> {
        (self.func)(state).await
    }
}

// ---------------------------------------------------------------------------
// AgentExecutor
// ---------------------------------------------------------------------------

/// An executor that runs an [`Agent`] as a workflow node.
///
/// Reads input messages from a state key, runs the agent, and writes the
/// response text (and full response) to output keys.
///
/// Mirrors .NET's `AgentExecutor` and Python's `AgentExecutor`.
pub struct AgentExecutor {
    id: String,
    agent: Arc<dyn Agent>,
    /// State key to read input messages from (as JSON array of Message).
    input_key: String,
    /// State key to write the response text to.
    output_key: String,
    /// State key to write the full AgentResponse JSON to.
    response_key: Option<String>,
}

impl AgentExecutor {
    pub fn new(agent: Arc<dyn Agent>) -> Self {
        let id = agent.id().to_string();
        Self {
            id,
            agent,
            input_key: "input".to_string(),
            output_key: "output".to_string(),
            response_key: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_input_key(mut self, key: impl Into<String>) -> Self {
        self.input_key = key.into();
        self
    }

    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    pub fn with_response_key(mut self, key: impl Into<String>) -> Self {
        self.response_key = Some(key.into());
        self
    }
}

#[async_trait]
impl Executor for AgentExecutor {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, state: &WorkflowState) -> AgentResult<()> {
        // Read input from state.
        let input: Vec<Message> = match state.get(&self.input_key).await {
            Some(val) => {
                serde_json::from_value::<Vec<Message>>(val.clone()).unwrap_or_else(|_| {
                    // If it's a plain string, treat it as a single user message.
                    let text = val.as_str().unwrap_or("").to_string();
                    vec![Message::user(text)]
                })
            }
            None => vec![],
        };

        let mut session = AgentSession::new();
        let response = self.agent.run(input, &mut session, None).await?;

        // Write outputs.
        state
            .set(&self.output_key, serde_json::json!(response.text))
            .await;

        if let Some(key) = &self.response_key {
            state
                .set(key, serde_json::to_value(&response).unwrap_or_default())
                .await;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// WorkflowExecutor (sub-workflow)
// ---------------------------------------------------------------------------

/// An executor that runs a nested workflow as a single node.
///
/// Mirrors Python's sub-workflow composition.
pub struct WorkflowExecutor {
    id: String,
    workflow: Arc<crate::Workflow>,
}

impl WorkflowExecutor {
    pub fn new(id: impl Into<String>, workflow: Arc<crate::Workflow>) -> Self {
        Self {
            id: id.into(),
            workflow,
        }
    }
}

#[async_trait]
impl Executor for WorkflowExecutor {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, state: &WorkflowState) -> AgentResult<()> {
        // Share state with the sub-workflow.
        self.workflow.run_with_state(state).await
    }
}
