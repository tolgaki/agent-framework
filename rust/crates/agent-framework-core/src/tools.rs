// Copyright (c) Microsoft. All rights reserved.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::AgentResult;

/// The JSON Schema definition for a tool's parameters.
///
/// This is sent to the model so it knows how to call the tool. Corresponds to
/// Python's `ToolDefinition` / .NET's `AIFunctionMetadata`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// The tool's unique name (must match the pattern `[a-zA-Z0-9_-]+`).
    pub name: String,

    /// A human-readable description of what the tool does.
    pub description: String,

    /// A JSON Schema object describing the tool's input parameters.
    pub parameters_schema: serde_json::Value,
}

impl ToolDefinition {
    /// Create a new tool definition.
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters_schema: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters_schema,
        }
    }

    /// Create a tool definition with no parameters.
    pub fn no_params(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters_schema: serde_json::json!({
                "type": "object",
                "properties": {},
            }),
        }
    }
}

/// A tool that can be invoked by an agent during a conversation.
///
/// Corresponds to Python's `FunctionTool` and .NET's `AITool` / `AIFunction`.
///
/// # Implementing a Tool
///
/// ```rust
/// use async_trait::async_trait;
/// use agent_framework_core::tools::{FunctionTool, ToolDefinition};
/// use agent_framework_core::error::AgentResult;
///
/// struct WeatherTool;
///
/// #[async_trait]
/// impl FunctionTool for WeatherTool {
///     fn definition(&self) -> &ToolDefinition {
///         // In practice, store this in a field initialized at construction.
///         todo!()
///     }
///
///     async fn invoke(&self, args: serde_json::Value) -> AgentResult<serde_json::Value> {
///         let location = args["location"].as_str().unwrap_or("unknown");
///         Ok(serde_json::json!({ "temperature": 72, "location": location }))
///     }
/// }
/// ```
#[async_trait]
pub trait FunctionTool: Send + Sync {
    /// Return the tool's definition (name, description, parameter schema).
    fn definition(&self) -> &ToolDefinition;

    /// Invoke the tool with the given arguments.
    ///
    /// The `args` value matches the JSON Schema declared in [`definition()`](Self::definition).
    ///
    /// # Errors
    /// Returns [`AgentError::ToolError`](crate::error::AgentError::ToolError) on failure.
    async fn invoke(&self, args: serde_json::Value) -> AgentResult<serde_json::Value>;
}

/// A simple function tool built from a closure.
///
/// Use [`tool_fn`] to construct.
pub struct ClosureTool<F> {
    definition: ToolDefinition,
    func: F,
}

/// Create a [`FunctionTool`] from an async closure.
///
/// # Example
///
/// ```rust
/// use agent_framework_core::tools::{tool_fn, ToolDefinition};
///
/// let def = ToolDefinition::no_params("greet", "Say hello");
/// let tool = tool_fn(def, |_args| async {
///     Ok(serde_json::json!("Hello!"))
/// });
/// ```
pub fn tool_fn<F, Fut>(definition: ToolDefinition, func: F) -> ClosureTool<F>
where
    F: Fn(serde_json::Value) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = AgentResult<serde_json::Value>> + Send,
{
    ClosureTool { definition, func }
}

#[async_trait]
impl<F, Fut> FunctionTool for ClosureTool<F>
where
    F: Fn(serde_json::Value) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = AgentResult<serde_json::Value>> + Send,
{
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn invoke(&self, args: serde_json::Value) -> AgentResult<serde_json::Value> {
        (self.func)(args).await
    }
}
