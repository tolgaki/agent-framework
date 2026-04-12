// Copyright (c) Microsoft. All rights reserved.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::tools::ToolDefinition;

/// Serde helper that encodes `Vec<u8>` as a base64 string instead of a JSON
/// integer array.  Keeps binary payloads compact (4/3× vs ~3-4× for `[u8]`).
mod base64_serde {
    use base64::engine::{general_purpose::STANDARD, Engine};
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(data: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&STANDARD.encode(data))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let encoded = String::deserialize(d)?;
        STANDARD.decode(encoded).map_err(serde::de::Error::custom)
    }
}

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
    Data {
        #[serde(with = "base64_serde")]
        data: Vec<u8>,
        media_type: String,
    },

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

/// How the model should format its response.
///
/// Mirrors Python's `response_format` and .NET's `ChatResponseFormat`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Unconstrained text output (the default).
    Text,
    /// The model must respond with a syntactically valid JSON object.
    JsonObject,
    /// The model must respond with JSON conforming to the supplied schema.
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

/// How the model should choose whether to call tools.
///
/// Mirrors Python's `tool_choice` and .NET's `ChatToolMode`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// Let the model decide whether to call tools (default).
    Auto,
    /// Forbid tool calls this turn.
    None,
    /// Require the model to call at least one tool this turn.
    Required,
    /// Force the model to call this specific tool by name.
    Specific { name: String },
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

    /// Top-k sampling (Anthropic, Google, and some others).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Random seed for reproducible sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    /// Frequency penalty (-2.0 to 2.0). OpenAI-compatible providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Presence penalty (-2.0 to 2.0). OpenAI-compatible providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Logit bias: token id → bias value. OpenAI-compatible providers.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub logit_bias: HashMap<String, f32>,

    /// Stable end-user identifier for abuse monitoring (OpenAI's `user` field).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// Required response format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// How the model should choose whether to call tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,

    /// Stop sequences.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,

    /// Tools the model may call during this request.
    ///
    /// Mirrors `ChatOptions.Tools` in the .NET SDK and the `"tools"` key in the
    /// Python SDK's options dict.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,

    /// Additional provider-specific options that don't map to a typed field.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, serde_json::Value>,
}

impl ChatOptions {
    /// Merge `overrides` into a clone of `self`.
    ///
    /// For scalar/option fields, any `Some(..)` in `overrides` replaces the
    /// value from `self`. For collection fields (`stop_sequences`, `tools`,
    /// `logit_bias`, `extra`), a non-empty collection in `overrides` replaces
    /// the one in `self`.
    pub fn merge(&self, overrides: &ChatOptions) -> Self {
        let mut out = self.clone();
        if overrides.model.is_some() {
            out.model = overrides.model.clone();
        }
        if overrides.max_tokens.is_some() {
            out.max_tokens = overrides.max_tokens;
        }
        if overrides.temperature.is_some() {
            out.temperature = overrides.temperature;
        }
        if overrides.top_p.is_some() {
            out.top_p = overrides.top_p;
        }
        if overrides.top_k.is_some() {
            out.top_k = overrides.top_k;
        }
        if overrides.seed.is_some() {
            out.seed = overrides.seed;
        }
        if overrides.frequency_penalty.is_some() {
            out.frequency_penalty = overrides.frequency_penalty;
        }
        if overrides.presence_penalty.is_some() {
            out.presence_penalty = overrides.presence_penalty;
        }
        if !overrides.logit_bias.is_empty() {
            out.logit_bias = overrides.logit_bias.clone();
        }
        if overrides.user.is_some() {
            out.user = overrides.user.clone();
        }
        if overrides.response_format.is_some() {
            out.response_format = overrides.response_format.clone();
        }
        if overrides.tool_choice.is_some() {
            out.tool_choice = overrides.tool_choice.clone();
        }
        if !overrides.stop_sequences.is_empty() {
            out.stop_sequences = overrides.stop_sequences.clone();
        }
        if !overrides.tools.is_empty() {
            out.tools = overrides.tools.clone();
        }
        if !overrides.extra.is_empty() {
            out.extra = overrides.extra.clone();
        }
        out
    }
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

/// Per-call overrides for a single agent invocation.
///
/// Mirrors .NET's `AgentRunOptions` and Python's per-call kwargs. Any field
/// left unset inherits the agent's default configured at build time.
#[derive(Debug, Clone, Default)]
pub struct AgentRunOptions {
    /// Override individual [`ChatOptions`] fields for this call only. Only
    /// `Some(...)` fields take effect; `None` fields keep the agent's defaults.
    pub chat_options: Option<ChatOptions>,

    /// Additional instructions prepended (as an extra system message) to the
    /// agent's configured instructions for this call only.
    pub additional_instructions: Option<String>,
}

impl AgentRunOptions {
    /// Create an empty options struct (equivalent to [`Default::default`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set per-call [`ChatOptions`] overrides.
    pub fn with_chat_options(mut self, options: ChatOptions) -> Self {
        self.chat_options = Some(options);
        self
    }

    /// Add extra system-level instructions for this call only.
    pub fn with_additional_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.additional_instructions = Some(instructions.into());
        self
    }
}
