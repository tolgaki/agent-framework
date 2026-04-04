// Copyright (c) Microsoft. All rights reserved.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use tokio_stream::StreamExt;

use crate::error::AgentResult;
use crate::types::{AgentResponseUpdate, ChatResponseUpdate};

/// A stream of chat response updates from a provider.
///
/// Wraps an async stream of [`ChatResponseUpdate`] items. This corresponds to
/// Python's `ResponseStream` and .NET's streaming `IAsyncEnumerable<StreamingChatCompletionUpdate>`.
pub struct ResponseStream {
    inner: Pin<Box<dyn Stream<Item = AgentResult<ChatResponseUpdate>> + Send>>,
}

impl ResponseStream {
    /// Create a new response stream from any async stream of updates.
    pub fn new(stream: impl Stream<Item = AgentResult<ChatResponseUpdate>> + Send + 'static) -> Self {
        Self {
            inner: Box::pin(stream),
        }
    }
}

impl Stream for ResponseStream {
    type Item = AgentResult<ChatResponseUpdate>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// A stream of agent response updates.
///
/// Wraps a [`ResponseStream`] and maps updates to [`AgentResponseUpdate`].
pub struct AgentResponseStream {
    inner: Pin<Box<dyn Stream<Item = AgentResult<AgentResponseUpdate>> + Send>>,
}

impl AgentResponseStream {
    /// Create a new agent response stream from any async stream.
    pub fn new(stream: impl Stream<Item = AgentResult<AgentResponseUpdate>> + Send + 'static) -> Self {
        Self {
            inner: Box::pin(stream),
        }
    }

    /// Create an agent response stream from a chat response stream.
    pub fn from_response_stream(stream: ResponseStream) -> Self {
        let mapped = stream.map(|result| {
            result.map(|update| AgentResponseUpdate {
                text: update.text.clone(),
                inner: update,
            })
        });
        Self::new(mapped)
    }

    /// Collect all updates and concatenate text into a final string.
    pub async fn collect_text(self) -> AgentResult<String> {
        let mut text = String::new();
        let mut stream = self.inner;
        while let Some(update) = Pin::new(&mut stream).next().await {
            if let Some(t) = update?.text {
                text.push_str(&t);
            }
        }
        Ok(text)
    }
}

impl Stream for AgentResponseStream {
    type Item = AgentResult<AgentResponseUpdate>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}
