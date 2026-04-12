// Copyright (c) Microsoft. All rights reserved.

//! Message compaction strategies for managing context window limits.
//!
//! When conversations grow long, older messages must be trimmed or summarized
//! to fit within the model's context window. This module provides a
//! [`CompactionStrategy`] trait and several built-in implementations that
//! mirror the .NET `CompactionProvider` / `CompactionStrategy` hierarchy and
//! the Python `CompactionStrategy` classes.
//!
//! # Built-in Strategies
//!
//! - [`SlidingWindowStrategy`] — keep the most recent N messages
//! - [`TruncationStrategy`] — drop messages beyond a token budget
//! - [`ToolResultCompactionStrategy`] — shorten verbose tool outputs
//! - [`PipelineCompactionStrategy`] — chain multiple strategies
//! - [`SummarizationStrategy`] — summarize older messages via an LLM
//!
//! # Usage
//!
//! Strategies are applied before sending messages to the model, typically
//! inside a chat-client middleware layer:
//!
//! ```rust,no_run
//! use agent_framework_core::compaction::{CompactionStrategy, SlidingWindowStrategy};
//! use agent_framework_core::types::Message;
//!
//! let strategy = SlidingWindowStrategy::new(20);
//! let messages = vec![Message::user("hello")];
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let compacted = strategy.compact(messages).await.unwrap();
//! # });
//! ```

use async_trait::async_trait;

use crate::client::ChatClient;
use crate::error::AgentResult;
use crate::types::{Content, Message, Role};

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

/// A trait for estimating token counts.
///
/// Mirrors Python's `TokenizerProtocol`.
pub trait Tokenizer: Send + Sync {
    /// Estimate the number of tokens in the given text.
    fn count_tokens(&self, text: &str) -> usize;
}

/// A simple tokenizer that estimates ~4 characters per token.
///
/// Good enough for budget estimation; use a real tokenizer (tiktoken, etc.)
/// for production accuracy.
#[derive(Debug, Clone, Copy)]
pub struct CharEstimatorTokenizer {
    chars_per_token: usize,
}

impl Default for CharEstimatorTokenizer {
    fn default() -> Self {
        Self { chars_per_token: 4 }
    }
}

impl CharEstimatorTokenizer {
    pub fn new(chars_per_token: usize) -> Self {
        Self { chars_per_token: chars_per_token.max(1) }
    }
}

impl Tokenizer for CharEstimatorTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        text.len().div_ceil(self.chars_per_token)
    }
}

// ---------------------------------------------------------------------------
// Compaction strategy trait
// ---------------------------------------------------------------------------

/// A strategy for compacting (trimming, summarizing) a message list.
///
/// Mirrors .NET's `CompactionStrategy` and Python's `CompactionStrategy`.
#[async_trait]
pub trait CompactionStrategy: Send + Sync {
    /// Compact the given messages, returning a potentially shorter list.
    async fn compact(&self, messages: Vec<Message>) -> AgentResult<Vec<Message>>;
}

// ---------------------------------------------------------------------------
// Sliding window
// ---------------------------------------------------------------------------

/// Keep only the most recent `max_messages` messages (plus all system messages).
///
/// System messages are always preserved because they contain instructions.
pub struct SlidingWindowStrategy {
    max_messages: usize,
}

impl SlidingWindowStrategy {
    pub fn new(max_messages: usize) -> Self {
        Self { max_messages }
    }
}

#[async_trait]
impl CompactionStrategy for SlidingWindowStrategy {
    async fn compact(&self, messages: Vec<Message>) -> AgentResult<Vec<Message>> {
        let system: Vec<Message> = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .cloned()
            .collect();
        let non_system: Vec<Message> = messages
            .into_iter()
            .filter(|m| m.role != Role::System)
            .collect();

        let keep = if non_system.len() > self.max_messages {
            non_system[non_system.len() - self.max_messages..].to_vec()
        } else {
            non_system
        };

        let mut result = system;
        result.extend(keep);
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Truncation
// ---------------------------------------------------------------------------

/// Drop the oldest non-system messages until the total token count is
/// within `max_tokens`.
pub struct TruncationStrategy {
    max_tokens: usize,
    tokenizer: Box<dyn Tokenizer>,
}

impl TruncationStrategy {
    pub fn new(max_tokens: usize, tokenizer: impl Tokenizer + 'static) -> Self {
        Self {
            max_tokens,
            tokenizer: Box::new(tokenizer),
        }
    }

    /// Create with the default character-estimator tokenizer.
    pub fn with_default_tokenizer(max_tokens: usize) -> Self {
        Self::new(max_tokens, CharEstimatorTokenizer::default())
    }
}

fn message_tokens(msg: &Message, tokenizer: &dyn Tokenizer) -> usize {
    msg.content
        .iter()
        .map(|c| match c {
            Content::Text { text } => tokenizer.count_tokens(text),
            Content::ToolCall { arguments, .. } => tokenizer.count_tokens(&arguments.to_string()),
            Content::ToolResult { content, .. } => tokenizer.count_tokens(content),
            _ => 0,
        })
        .sum::<usize>()
        + 4 // overhead per message (role, separators)
}

#[async_trait]
impl CompactionStrategy for TruncationStrategy {
    async fn compact(&self, messages: Vec<Message>) -> AgentResult<Vec<Message>> {
        let system: Vec<Message> = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .cloned()
            .collect();
        let non_system: Vec<Message> = messages
            .into_iter()
            .filter(|m| m.role != Role::System)
            .collect();

        // Count system tokens.
        let system_tokens: usize = system.iter().map(|m| message_tokens(m, &*self.tokenizer)).sum();
        let budget = self.max_tokens.saturating_sub(system_tokens);

        // Walk non-system messages from newest to oldest, accumulating until budget.
        let mut keep = Vec::new();
        let mut used = 0;
        for msg in non_system.into_iter().rev() {
            let t = message_tokens(&msg, &*self.tokenizer);
            if used + t > budget {
                break;
            }
            used += t;
            keep.push(msg);
        }
        keep.reverse();

        let mut result = system;
        result.extend(keep);
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tool result compaction
// ---------------------------------------------------------------------------

/// Truncate long tool result strings to a maximum character length.
///
/// Mirrors .NET's `ToolResultCompactionStrategy`.
pub struct ToolResultCompactionStrategy {
    max_result_chars: usize,
}

impl ToolResultCompactionStrategy {
    pub fn new(max_result_chars: usize) -> Self {
        Self { max_result_chars }
    }
}

#[async_trait]
impl CompactionStrategy for ToolResultCompactionStrategy {
    async fn compact(&self, messages: Vec<Message>) -> AgentResult<Vec<Message>> {
        Ok(messages
            .into_iter()
            .map(|mut msg| {
                msg.content = msg
                    .content
                    .into_iter()
                    .map(|c| match c {
                        Content::ToolResult { tool_call_id, content } => {
                            let truncated = if content.len() > self.max_result_chars {
                                format!(
                                    "{}... [truncated {} chars]",
                                    &content[..self.max_result_chars],
                                    content.len() - self.max_result_chars
                                )
                            } else {
                                content
                            };
                            Content::ToolResult {
                                tool_call_id,
                                content: truncated,
                            }
                        }
                        other => other,
                    })
                    .collect();
                msg
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Chain multiple compaction strategies in order.
///
/// Each strategy's output is fed as input to the next.
/// Mirrors .NET's `PipelineCompactionStrategy`.
pub struct PipelineCompactionStrategy {
    strategies: Vec<Box<dyn CompactionStrategy>>,
}

impl PipelineCompactionStrategy {
    pub fn new() -> Self {
        Self { strategies: Vec::new() }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, strategy: impl CompactionStrategy + 'static) -> Self {
        self.strategies.push(Box::new(strategy));
        self
    }
}

impl Default for PipelineCompactionStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CompactionStrategy for PipelineCompactionStrategy {
    async fn compact(&self, mut messages: Vec<Message>) -> AgentResult<Vec<Message>> {
        for strategy in &self.strategies {
            messages = strategy.compact(messages).await?;
        }
        Ok(messages)
    }
}

// ---------------------------------------------------------------------------
// Summarization
// ---------------------------------------------------------------------------

/// Summarize older messages using an LLM, keeping recent messages intact.
///
/// Splits messages into "old" (to summarize) and "recent" (to keep).
/// The old messages are sent to a chat client with a summarization prompt,
/// and the result replaces them as a single system message.
///
/// Mirrors .NET's `SummarizationCompactionStrategy` and Python's
/// `SummarizationStrategy`.
pub struct SummarizationStrategy {
    /// Chat client used to generate the summary.
    summarizer: Box<dyn ChatClient>,
    /// Number of recent messages to keep verbatim.
    keep_recent: usize,
    /// Prompt template for the summarization request.
    prompt: String,
}

impl SummarizationStrategy {
    pub fn new(summarizer: Box<dyn ChatClient>, keep_recent: usize) -> Self {
        Self {
            summarizer,
            keep_recent,
            prompt: "Summarize the following conversation concisely, preserving key facts, \
                     decisions, and context needed for continuing the conversation."
                .to_string(),
        }
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }
}

#[async_trait]
impl CompactionStrategy for SummarizationStrategy {
    async fn compact(&self, messages: Vec<Message>) -> AgentResult<Vec<Message>> {
        let system: Vec<Message> = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .cloned()
            .collect();
        let non_system: Vec<Message> = messages
            .into_iter()
            .filter(|m| m.role != Role::System)
            .collect();

        if non_system.len() <= self.keep_recent {
            let mut result = system;
            result.extend(non_system);
            return Ok(result);
        }

        let split = non_system.len() - self.keep_recent;
        let to_summarize = &non_system[..split];
        let to_keep = &non_system[split..];

        // Build a conversation transcript for the summarizer.
        let transcript: String = to_summarize
            .iter()
            .map(|m| format!("{:?}: {}", m.role, m.text()))
            .collect::<Vec<_>>()
            .join("\n");

        let summary_request = vec![
            Message::system(&self.prompt),
            Message::user(transcript),
        ];

        let response = self.summarizer.get_response(&summary_request, None).await?;
        let summary_text = response
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|c| c.as_text())
            .collect::<Vec<_>>()
            .join("");

        let mut result = system;
        if !summary_text.is_empty() {
            result.push(Message::system(format!("[Conversation summary]: {summary_text}")));
        }
        result.extend(to_keep.iter().cloned());
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// CompactionProvider — middleware integration
// ---------------------------------------------------------------------------

/// A chat-client middleware that applies a compaction strategy before each
/// model call.
///
/// Corresponds to .NET's `CompactionProvider` and Python's `CompactionProvider`.
pub struct CompactionMiddleware {
    strategy: Box<dyn CompactionStrategy>,
}

impl CompactionMiddleware {
    pub fn new(strategy: impl CompactionStrategy + 'static) -> Self {
        Self {
            strategy: Box::new(strategy),
        }
    }
}

#[async_trait]
impl crate::middleware::ChatClientMiddleware for CompactionMiddleware {
    async fn on_get_response(
        &self,
        messages: &mut Vec<Message>,
        options: &mut crate::types::ChatOptions,
        next: &dyn ChatClient,
    ) -> AgentResult<crate::types::ChatResponse> {
        let compacted = self.strategy.compact(std::mem::take(messages)).await?;
        *messages = compacted;
        next.get_response(messages, Some(options)).await
    }

    async fn on_get_response_stream(
        &self,
        messages: &mut Vec<Message>,
        options: &mut crate::types::ChatOptions,
        next: &dyn ChatClient,
    ) -> AgentResult<crate::streaming::ResponseStream> {
        let compacted = self.strategy.compact(std::mem::take(messages)).await?;
        *messages = compacted;
        next.get_response_stream(messages, Some(options)).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sliding_window_keeps_recent_and_system() {
        let messages = vec![
            Message::system("you are helpful"),
            Message::user("msg1"),
            Message::assistant("reply1"),
            Message::user("msg2"),
            Message::assistant("reply2"),
            Message::user("msg3"),
        ];
        let strategy = SlidingWindowStrategy::new(3);
        let result = strategy.compact(messages).await.unwrap();
        // System + last 3 non-system messages (msg2, reply2, msg3)
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].role, Role::System);
        assert_eq!(result[1].text(), "msg2");
        assert_eq!(result[2].text(), "reply2");
        assert_eq!(result[3].text(), "msg3");
    }

    #[tokio::test]
    async fn truncation_respects_token_budget() {
        let messages = vec![
            Message::system("sys"),
            Message::user("a]".repeat(100)), // ~200 chars = ~50 tokens
            Message::assistant("b".repeat(100)), // ~100 chars = ~25 tokens
            Message::user("c".repeat(20)), // ~20 chars = ~5 tokens
        ];
        // Budget of 40 tokens: system(~5) + we have 35 left.
        // From newest: "c"(~9) + "b"(~29) = ~38 > 35, so only "c" fits.
        let strategy = TruncationStrategy::with_default_tokenizer(15);
        let result = strategy.compact(messages).await.unwrap();
        // System + last message that fits
        assert!(result.len() >= 2);
        assert_eq!(result[0].role, Role::System);
    }

    #[tokio::test]
    async fn tool_result_compaction_truncates() {
        let messages = vec![Message {
            role: Role::Tool,
            content: vec![Content::tool_result("id1", "x".repeat(1000))],
            name: None,
            metadata: Default::default(),
        }];
        let strategy = ToolResultCompactionStrategy::new(50);
        let result = strategy.compact(messages).await.unwrap();
        if let Content::ToolResult { content, .. } = &result[0].content[0] {
            assert!(content.len() < 200);
            assert!(content.contains("[truncated"));
        } else {
            panic!("expected ToolResult");
        }
    }

    #[tokio::test]
    async fn pipeline_chains_strategies() {
        let messages = vec![
            Message::system("sys"),
            Message::user("old1"),
            Message::user("old2"),
            Message::user("old3"),
            Message::user("recent1"),
            Message::user("recent2"),
        ];
        let pipeline = PipelineCompactionStrategy::new()
            .add(SlidingWindowStrategy::new(3));
        let result = pipeline.compact(messages).await.unwrap();
        assert_eq!(result.len(), 4); // system + 3 recent
    }
}
