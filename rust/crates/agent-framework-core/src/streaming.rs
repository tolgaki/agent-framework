// Copyright (c) Microsoft. All rights reserved.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use tokio_stream::StreamExt;

use crate::error::AgentResult;
use crate::types::{AgentResponseUpdate, ChatResponseUpdate};

/// RAII guard that aborts a spawned task when dropped.
///
/// Providers that spawn background SSE readers use this to ensure that
/// dropping a [`ResponseStream`] mid-way through aborts the reader task
/// immediately, rather than letting it run until the HTTP request times out.
pub struct AbortOnDrop(tokio::task::AbortHandle);

impl AbortOnDrop {
    /// Wrap an abort handle so the task is aborted on drop.
    pub fn new(handle: tokio::task::AbortHandle) -> Self {
        Self(handle)
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Stream wrapper that holds an [`AbortOnDrop`] guard alongside an inner stream.
///
/// When this stream is dropped, the guard aborts the associated task.
pub struct AbortingStream<S> {
    inner: S,
    _guard: AbortOnDrop,
}

impl<S> AbortingStream<S> {
    pub fn new(inner: S, guard: AbortOnDrop) -> Self {
        Self { inner, _guard: guard }
    }
}

impl<S> Stream for AbortingStream<S>
where
    S: Stream + Unpin,
{
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.get_mut().inner).poll_next(cx)
    }
}

/// A stream of chat response updates from a provider.
///
/// Wraps an async stream of [`ChatResponseUpdate`] items. Corresponds to Python's
/// `ResponseStream` and .NET's `IAsyncEnumerable<ChatResponseUpdate>`.
///
/// Provider streams are `'static`: they don't hold borrows into the caller and
/// can be moved/dropped freely.
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
/// Unlike [`ResponseStream`], agent streams may borrow from the agent and the
/// caller's session while they are being driven, so they carry a lifetime.
/// Use `'static` if you don't need to borrow.
pub struct AgentResponseStream<'a> {
    inner: Pin<Box<dyn Stream<Item = AgentResult<AgentResponseUpdate>> + Send + 'a>>,
}

impl<'a> AgentResponseStream<'a> {
    /// Create a new agent response stream from any async stream.
    pub fn new(stream: impl Stream<Item = AgentResult<AgentResponseUpdate>> + Send + 'a) -> Self {
        Self {
            inner: Box::pin(stream),
        }
    }

    /// Create an agent response stream from a chat response stream.
    pub fn from_response_stream(stream: ResponseStream) -> AgentResponseStream<'static> {
        let mapped = stream.map(|result| {
            result.map(|update| AgentResponseUpdate {
                text: update.text.clone(),
                inner: update,
            })
        });
        AgentResponseStream::new(mapped)
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

impl<'a> Stream for AgentResponseStream<'a> {
    type Item = AgentResult<AgentResponseUpdate>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}
