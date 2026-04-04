// Copyright (c) Microsoft. All rights reserved.

/// Errors that can occur during agent operations.
#[derive(thiserror::Error, Debug)]
pub enum AgentError {
    /// The request to the agent was invalid.
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// The agent produced an invalid response.
    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    /// An error occurred in the LLM provider.
    #[error("Provider error: {message}")]
    ProviderError { message: String, status_code: Option<u16> },

    /// An error occurred during tool invocation.
    #[error("Tool invocation error for '{tool_name}': {message}")]
    ToolError { tool_name: String, message: String },

    /// A serialization or deserialization error.
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// An HTTP transport error.
    #[error("HTTP error: {0}")]
    HttpError(String),

    /// A generic boxed error for extensibility.
    #[error("{0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl AgentError {
    /// Create a provider error with an optional HTTP status code.
    pub fn provider(message: impl Into<String>, status_code: Option<u16>) -> Self {
        Self::ProviderError {
            message: message.into(),
            status_code,
        }
    }

    /// Create a tool invocation error.
    pub fn tool(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ToolError {
            tool_name: name.into(),
            message: message.into(),
        }
    }
}

/// A specialized [`Result`] type for agent operations.
pub type AgentResult<T> = Result<T, AgentError>;
