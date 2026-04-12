// Copyright (c) Microsoft. All rights reserved.

use async_trait::async_trait;

use crate::error::AgentResult;
use crate::streaming::ResponseStream;
use crate::types::{ChatOptions, ChatResponse, Message};

/// A chat client that can communicate with an LLM provider.
///
/// This is the core provider abstraction, corresponding to Python's
/// `SupportsChatGetResponse` protocol and .NET's `IChatClient` interface.
///
/// Implementors translate between the framework's unified message types and
/// a specific provider's API (Anthropic, OpenAI, Azure, etc.).
#[async_trait]
pub trait ChatClient: Send + Sync {
    /// Send messages and return a complete response.
    ///
    /// # Arguments
    /// * `messages` - The conversation history to send.
    /// * `options` - Optional request parameters (model, temperature, etc.).
    ///
    /// # Errors
    /// Returns [`AgentError`](crate::error::AgentError) on provider or transport failures.
    async fn get_response(&self, messages: &[Message], options: Option<&ChatOptions>) -> AgentResult<ChatResponse>;

    /// Send messages and return a streaming response.
    ///
    /// # Arguments
    /// * `messages` - The conversation history to send.
    /// * `options` - Optional request parameters.
    ///
    /// # Errors
    /// Returns [`AgentError`](crate::error::AgentError) if the stream cannot be established.
    async fn get_response_stream(
        &self,
        messages: &[Message],
        options: Option<&ChatOptions>,
    ) -> AgentResult<ResponseStream>;
}
