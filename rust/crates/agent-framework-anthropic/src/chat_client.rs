// Copyright (c) Microsoft. All rights reserved.

use std::collections::HashMap;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use tracing::debug;

use agent_framework_core::client::ChatClient;
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::streaming::ResponseStream;
use agent_framework_core::types::{ChatOptions, ChatResponse, Content, FinishReason, Message, Role, Usage};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const DEFAULT_API_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Configuration for the Anthropic chat client.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    /// Anthropic API key.
    pub api_key: String,

    /// Model identifier (e.g., "claude-sonnet-4-20250514").
    pub model: String,

    /// Maximum tokens to generate. Defaults to 4096.
    pub max_tokens: u32,

    /// Base URL for the API. Defaults to `https://api.anthropic.com`.
    pub base_url: String,
}

impl AnthropicConfig {
    /// Create a config from environment variables.
    ///
    /// Reads `ANTHROPIC_API_KEY` and optionally `ANTHROPIC_MODEL`.
    pub fn from_env() -> AgentResult<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| AgentError::InvalidRequest("ANTHROPIC_API_KEY environment variable is not set".to_string()))?;
        let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string());

        Ok(Self {
            api_key,
            model,
            max_tokens: DEFAULT_MAX_TOKENS,
            base_url: DEFAULT_BASE_URL.to_string(),
        })
    }
}

/// A [`ChatClient`] implementation for the Anthropic Messages API.
pub struct AnthropicChatClient {
    config: AnthropicConfig,
    http: reqwest::Client,
}

impl AnthropicChatClient {
    /// Create a new Anthropic chat client.
    pub fn new(config: AnthropicConfig) -> Self {
        let http = reqwest::Client::new();
        Self { config, http }
    }

    /// Build the request body for the Anthropic Messages API.
    fn build_request_body(&self, messages: &[Message], options: Option<&ChatOptions>) -> AgentResult<AnthropicRequest> {
        let model = options.and_then(|o| o.model.as_deref()).unwrap_or(&self.config.model);

        let max_tokens = options.and_then(|o| o.max_tokens).unwrap_or(self.config.max_tokens);

        // Extract system messages.
        let system: Option<String> = {
            let system_texts: Vec<&str> = messages
                .iter()
                .filter(|m| m.role == Role::System)
                .flat_map(|m| m.content.iter())
                .filter_map(|c| c.as_text())
                .collect();
            if system_texts.is_empty() {
                None
            } else {
                Some(system_texts.join("\n\n"))
            }
        };

        // Convert non-system messages to Anthropic format.
        let api_messages: Vec<AnthropicMessage> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(message_to_anthropic)
            .collect();

        // Extract tool definitions if provided in extras.
        let tools: Option<Vec<AnthropicTool>> = options
            .and_then(|o| o.extra.get("_tools"))
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .map(|defs: Vec<agent_framework_core::tools::ToolDefinition>| {
                defs.into_iter()
                    .map(|d| AnthropicTool {
                        name: d.name,
                        description: d.description,
                        input_schema: d.parameters_schema,
                    })
                    .collect()
            });

        Ok(AnthropicRequest {
            model: model.to_string(),
            max_tokens,
            system,
            messages: api_messages,
            tools,
            temperature: options.and_then(|o| o.temperature),
            top_p: options.and_then(|o| o.top_p),
            stop_sequences: options.map(|o| o.stop_sequences.clone()).filter(|s| !s.is_empty()),
        })
    }
}

#[async_trait]
impl ChatClient for AnthropicChatClient {
    async fn get_response(&self, messages: &[Message], options: Option<&ChatOptions>) -> AgentResult<ChatResponse> {
        let body = self.build_request_body(messages, options)?;
        let url = format!("{}/v1/messages", self.config.base_url);

        debug!(model = %body.model, "Sending request to Anthropic");

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.config.api_key)
                .map_err(|e| AgentError::InvalidRequest(format!("Invalid API key: {e}")))?,
        );
        headers.insert("anthropic-version", HeaderValue::from_static(DEFAULT_API_VERSION));

        let response = self
            .http
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentError::HttpError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(AgentError::provider(
                format!("Anthropic API error ({status}): {error_text}"),
                Some(status.as_u16()),
            ));
        }

        let api_response: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| AgentError::HttpError(e.to_string()))?;

        Ok(anthropic_response_to_chat_response(api_response))
    }

    fn get_response_stream(
        &self,
        _messages: &[Message],
        _options: Option<&ChatOptions>,
    ) -> AgentResult<ResponseStream> {
        // SSE streaming will be implemented in a future iteration.
        Err(AgentError::InvalidRequest(
            "Streaming not yet implemented for AnthropicChatClient".to_string(),
        ))
    }
}

// ---- Anthropic API types ----

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
enum AnthropicContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult { tool_use_id: String, content: String },
}

#[derive(Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    id: String,
    content: Vec<AnthropicContent>,
    stop_reason: Option<String>,
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

// ---- Conversion functions ----

fn message_to_anthropic(msg: &Message) -> AnthropicMessage {
    let role = match msg.role {
        Role::User | Role::Tool => "user",
        Role::Assistant => "assistant",
        Role::System => "user", // Should not reach here (filtered above).
    };

    let content: Vec<AnthropicContent> = msg
        .content
        .iter()
        .map(|c| match c {
            Content::Text { text } => AnthropicContent::Text { text: text.clone() },
            Content::ToolCall { id, name, arguments } => AnthropicContent::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: arguments.clone(),
            },
            Content::ToolResult { tool_call_id, content } => AnthropicContent::ToolResult {
                tool_use_id: tool_call_id.clone(),
                content: content.clone(),
            },
            Content::Data { media_type, .. } => AnthropicContent::Text {
                text: format!("[binary data: {media_type}]"),
            },
            Content::Uri { uri, .. } => AnthropicContent::Text {
                text: format!("[uri: {uri}]"),
            },
        })
        .collect();

    AnthropicMessage {
        role: role.to_string(),
        content,
    }
}

fn anthropic_response_to_chat_response(resp: AnthropicResponse) -> ChatResponse {
    let mut content_items = Vec::new();

    for block in &resp.content {
        match block {
            AnthropicContent::Text { text } => {
                content_items.push(Content::text(text));
            }
            AnthropicContent::ToolUse { id, name, input } => {
                content_items.push(Content::tool_call(id, name, input.clone()));
            }
            AnthropicContent::ToolResult { .. } => {
                // Tool results in response are unusual; skip.
            }
        }
    }

    let finish_reason = resp.stop_reason.as_deref().map(|r| match r {
        "end_turn" | "stop" => FinishReason::Stop,
        "max_tokens" => FinishReason::MaxTokens,
        "tool_use" => FinishReason::ToolUse,
        _ => FinishReason::Stop,
    });

    let usage = resp.usage.map(|u| Usage {
        input_tokens: u.input_tokens,
        output_tokens: u.output_tokens,
    });

    let message = Message {
        role: Role::Assistant,
        content: content_items,
        name: None,
        metadata: HashMap::new(),
    };

    ChatResponse {
        messages: vec![message],
        response_id: Some(resp.id),
        finish_reason,
        usage,
    }
}
