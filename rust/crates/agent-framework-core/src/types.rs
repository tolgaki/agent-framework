// Copyright (c) Microsoft. All rights reserved.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// The role of a message participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// A system-level instruction message.
    System,
    /// A user-provided message.
    User,
    /// An assistant (model) response.
    Assistant,
    /// A tool result message.
    Tool,
}

/// A piece of content within a message.
///
/// This unified content model mirrors the Python SDK's `Content` class and the
/// .NET SDK's `AIContent` hierarchy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    /// Plain text content.
    Text { text: String },

    /// Binary data content with a media type.
    Data { data: Vec<u8>, media_type: String },

    /// A URI reference to external content.
    Uri {
        uri: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
    },

    /// A tool/function call requested by the model.
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },

    /// The result of a tool/function invocation.
    ToolResult { tool_call_id: String, content: String },
}

impl Content {
    /// Create a text content item.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create a tool call content item.
    pub fn tool_call(id: impl Into<String>, name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    /// Create a tool result content item.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
        }
    }

    /// Extract text from this content, if it is a text variant.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }

    /// Extract the tool call from this content, if it is a tool call variant.
    pub fn as_tool_call(&self) -> Option<(&str, &str, &serde_json::Value)> {
        match self {
            Self::ToolCall { id, name, arguments } => Some((id, name, arguments)),
            _ => None,
        }
    }
}

/// A message in a conversation.
///
/// Corresponds to Python's `Message` and .NET's `ChatMessage`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message author.
    pub role: Role,

    /// The content blocks in this message.
    pub content: Vec<Content>,

    /// An optional name for the participant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Arbitrary metadata attached to this message.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Message {
    /// Create a user message with text content.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![Content::text(text)],
            name: None,
            metadata: HashMap::new(),
        }
    }

    /// Create an assistant message with text content.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![Content::text(text)],
            name: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a system message with text content.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![Content::text(text)],
            name: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a tool result message.
    pub fn tool_result(tool_call_id: impl Into<String>, result: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: vec![Content::tool_result(tool_call_id, result)],
            name: None,
            metadata: HashMap::new(),
        }
    }

    /// Extract all text content from this message, concatenated.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| c.as_text())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Extract all tool calls from this message.
    pub fn tool_calls(&self) -> Vec<(&str, &str, &serde_json::Value)> {
        self.content.iter().filter_map(|c| c.as_tool_call()).collect()
    }
}

/// Token usage information from a model response.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the input/prompt.
    #[serde(default)]
    pub input_tokens: u32,

    /// Number of tokens in the output/completion.
    #[serde(default)]
    pub output_tokens: u32,
}

/// Reason the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model reached a natural stop point.
    Stop,
    /// The model hit the maximum token limit.
    MaxTokens,
    /// The model wants to call one or more tools.
    ToolUse,
    /// The response was filtered by content policy.
    ContentFilter,
}

/// Options for a chat completion request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatOptions {
    /// The model identifier to use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// Maximum number of tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Sampling temperature (0.0 - 2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Nucleus sampling top-p (0.0 - 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Stop sequences.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,

    /// Additional provider-specific options.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, serde_json::Value>,
}

/// A complete response from a chat client.
///
/// Corresponds to Python's `ChatResponse` and .NET's `ChatCompletion`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The response messages from the model.
    pub messages: Vec<Message>,

    /// A provider-assigned response identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,

    /// The reason the model stopped generating.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Token usage statistics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// An incremental update during streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponseUpdate {
    /// Incremental text content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// A tool call being streamed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call: Option<ToolCallUpdate>,

    /// The finish reason, if this is the final update.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Usage info, typically sent with the final update.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// An incremental tool call update during streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallUpdate {
    /// The tool call ID.
    pub id: String,
    /// The tool name.
    pub name: String,
    /// Partial JSON arguments accumulated so far.
    pub arguments_delta: String,
}

/// A complete response from an agent run.
///
/// Wraps the underlying `ChatResponse` with convenience accessors.
/// Corresponds to Python's `AgentResponse` and .NET's `AgentResponse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    /// All messages produced during the agent run (including tool calls and results).
    pub messages: Vec<Message>,

    /// The final text output from the agent.
    pub text: String,

    /// The reason the model stopped generating.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,

    /// Token usage accumulated across all model calls in this run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl AgentResponse {
    /// Build an `AgentResponse` from the final model response and accumulated messages.
    pub fn from_chat_response(chat_response: ChatResponse, all_messages: Vec<Message>) -> Self {
        let text = chat_response
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|c| c.as_text())
            .collect::<Vec<_>>()
            .join("");

        Self {
            messages: all_messages,
            text,
            finish_reason: chat_response.finish_reason,
            usage: chat_response.usage,
        }
    }
}

/// An incremental update from an agent run (for streaming).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponseUpdate {
    /// Incremental text, if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// The underlying chat response update.
    pub inner: ChatResponseUpdate,
}
