// Copyright (c) Microsoft. All rights reserved.

use std::collections::HashMap;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use tracing::debug;

use agent_framework_core::client::ChatClient;
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::streaming::ResponseStream;
use agent_framework_core::types::{ChatOptions, ChatResponse, Content, FinishReason, Message, Role, Usage};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Configuration for the OpenAI chat client.
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    /// OpenAI API key.
    pub api_key: String,

    /// Model identifier (e.g., "gpt-4o").
    pub model: String,

    /// Maximum tokens to generate.
    pub max_tokens: u32,

    /// Base URL for the API.
    pub base_url: String,
}

impl OpenAIConfig {
    /// Create a config from environment variables.
    ///
    /// Reads `OPENAI_API_KEY` and optionally `OPENAI_MODEL`.
    pub fn from_env() -> AgentResult<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| AgentError::InvalidRequest("OPENAI_API_KEY environment variable is not set".to_string()))?;
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());

        Ok(Self {
            api_key,
            model,
            max_tokens: DEFAULT_MAX_TOKENS,
            base_url: DEFAULT_BASE_URL.to_string(),
        })
    }
}

/// A [`ChatClient`] implementation for the OpenAI Chat Completions API.
pub struct OpenAIChatClient {
    config: OpenAIConfig,
    http: reqwest::Client,
}

impl OpenAIChatClient {
    /// Create a new OpenAI chat client.
    pub fn new(config: OpenAIConfig) -> Self {
        Self {
            config,
            http: reqwest::Client::new(),
        }
    }

    fn build_request_body(&self, messages: &[Message], options: Option<&ChatOptions>) -> OpenAIRequest {
        let model = options
            .and_then(|o| o.model.as_deref())
            .unwrap_or(&self.config.model)
            .to_string();

        let max_tokens = options.and_then(|o| o.max_tokens).unwrap_or(self.config.max_tokens);

        let api_messages: Vec<OpenAIMessage> = messages.iter().map(message_to_openai).collect();

        // Extract tool definitions.
        let tools: Option<Vec<OpenAITool>> = options
            .and_then(|o| o.extra.get("_tools"))
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .map(|defs: Vec<agent_framework_core::tools::ToolDefinition>| {
                defs.into_iter()
                    .map(|d| OpenAITool {
                        r#type: "function".to_string(),
                        function: OpenAIFunction {
                            name: d.name,
                            description: d.description,
                            parameters: d.parameters_schema,
                        },
                    })
                    .collect()
            });

        OpenAIRequest {
            model,
            max_tokens: Some(max_tokens),
            messages: api_messages,
            tools,
            temperature: options.and_then(|o| o.temperature),
            top_p: options.and_then(|o| o.top_p),
            stop: options.map(|o| o.stop_sequences.clone()).filter(|s| !s.is_empty()),
        }
    }
}

#[async_trait]
impl ChatClient for OpenAIChatClient {
    async fn get_response(&self, messages: &[Message], options: Option<&ChatOptions>) -> AgentResult<ChatResponse> {
        let body = self.build_request_body(messages, options);
        let url = format!("{}/chat/completions", self.config.base_url);

        debug!(model = %body.model, "Sending request to OpenAI");

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.config.api_key))
                .map_err(|e| AgentError::InvalidRequest(format!("Invalid API key: {e}")))?,
        );

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
                format!("OpenAI API error ({status}): {error_text}"),
                Some(status.as_u16()),
            ));
        }

        let api_response: OpenAIResponse = response
            .json()
            .await
            .map_err(|e| AgentError::HttpError(e.to_string()))?;

        Ok(openai_response_to_chat_response(api_response))
    }

    fn get_response_stream(
        &self,
        _messages: &[Message],
        _options: Option<&ChatOptions>,
    ) -> AgentResult<ResponseStream> {
        Err(AgentError::InvalidRequest(
            "Streaming not yet implemented for OpenAIChatClient".to_string(),
        ))
    }
}

// ---- OpenAI API types ----

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Serialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Deserialize, Clone)]
struct OpenAIToolCall {
    id: String,
    r#type: String,
    function: OpenAIFunctionCall,
}

#[derive(Serialize, Deserialize, Clone)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Serialize)]
struct OpenAITool {
    r#type: String,
    function: OpenAIFunction,
}

#[derive(Serialize)]
struct OpenAIFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    id: String,
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIResponseMessage {
    #[allow(dead_code)]
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

// ---- Conversion functions ----

fn message_to_openai(msg: &Message) -> OpenAIMessage {
    let role = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    };

    // Check for tool calls in the message.
    let tool_calls: Vec<_> = msg
        .content
        .iter()
        .filter_map(|c| {
            if let Content::ToolCall { id, name, arguments } = c {
                Some(OpenAIToolCall {
                    id: id.clone(),
                    r#type: "function".to_string(),
                    function: OpenAIFunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(arguments).unwrap_or_default(),
                    },
                })
            } else {
                None
            }
        })
        .collect();

    // Check for tool result.
    let tool_call_id = msg.content.iter().find_map(|c| {
        if let Content::ToolResult { tool_call_id, .. } = c {
            Some(tool_call_id.clone())
        } else {
            None
        }
    });

    let text = msg.text();
    let content = if text.is_empty() { None } else { Some(text) };

    OpenAIMessage {
        role: role.to_string(),
        content,
        tool_calls: if tool_calls.is_empty() { None } else { Some(tool_calls) },
        tool_call_id,
    }
}

fn openai_response_to_chat_response(resp: OpenAIResponse) -> ChatResponse {
    let choice = resp.choices.into_iter().next();

    let (message, finish_reason_str) = match choice {
        Some(c) => (c.message, c.finish_reason),
        None => {
            return ChatResponse {
                messages: vec![],
                response_id: Some(resp.id),
                finish_reason: None,
                usage: None,
            };
        }
    };

    let mut content_items = Vec::new();

    if let Some(text) = message.content {
        content_items.push(Content::text(text));
    }

    if let Some(tool_calls) = message.tool_calls {
        for tc in tool_calls {
            let args: serde_json::Value =
                serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::Value::Null);
            content_items.push(Content::tool_call(&tc.id, &tc.function.name, args));
        }
    }

    let finish_reason = finish_reason_str.as_deref().map(|r| match r {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::MaxTokens,
        "tool_calls" => FinishReason::ToolUse,
        "content_filter" => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    });

    let usage = resp.usage.map(|u| Usage {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
    });

    let msg = Message {
        role: Role::Assistant,
        content: content_items,
        name: None,
        metadata: HashMap::new(),
    };

    ChatResponse {
        messages: vec![msg],
        response_id: Some(resp.id),
        finish_reason,
        usage,
    }
}
