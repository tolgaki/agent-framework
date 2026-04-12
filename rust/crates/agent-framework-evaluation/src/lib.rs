// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework Evaluation
//!
//! Provider-agnostic evaluation framework for testing AI agents.
//!
//! This crate mirrors the Python SDK's `_evaluation.py` module. It provides:
//!
//! - [`Evaluator`] trait for pluggable evaluation logic.
//! - [`EvalItem`] / [`EvalScore`] / [`EvalResults`] types for structuring evaluations.
//! - [`evaluate_agent`] orchestration function that runs items through an agent and evaluators.
//! - Built-in evaluators: [`KeywordCheckEvaluator`], [`ToolCalledEvaluator`],
//!   [`ResponseNotEmptyEvaluator`].
//! - [`ClosureEvaluator`] for wrapping async closures as evaluators.
//!
//! # Example
//!
//! ```rust,no_run
//! use agent_framework_evaluation::{
//!     EvalItem, EvalResults, KeywordCheckEvaluator, ResponseNotEmptyEvaluator,
//!     evaluate_agent,
//! };
//! # async fn example(agent: &dyn agent_framework_core::agent::Agent) {
//! let items = vec![
//!     EvalItem::new("What is the weather?")
//!         .with_expected_response("sunny"),
//! ];
//! let evaluators: Vec<Box<dyn agent_framework_evaluation::Evaluator>> = vec![
//!     Box::new(KeywordCheckEvaluator::new(vec!["sunny".into()])),
//!     Box::new(ResponseNotEmptyEvaluator),
//! ];
//! let results = evaluate_agent(agent, items, &evaluators).await.unwrap();
//! assert!(results.all_passed());
//! # }
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use agent_framework_core::agent::Agent;
use agent_framework_core::error::AgentResult;
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::{AgentResponse, Message};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A tool call that an agent is expected to make during evaluation.
///
/// Corresponds to Python's `ExpectedToolCall`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedToolCall {
    /// The tool/function name (e.g. `"get_weather"`).
    pub name: String,
    /// Expected arguments. `None` means "don't check arguments".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,
}

impl ExpectedToolCall {
    /// Create an expected tool call that only checks the tool name.
    pub fn named(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            args: None,
        }
    }

    /// Create an expected tool call that also checks arguments.
    pub fn with_args(name: impl Into<String>, args: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            args: Some(args),
        }
    }
}

/// A single item to be evaluated.
///
/// Represents one query that will be sent to the agent, along with optional
/// expected outputs for scoring. Corresponds to Python's `EvalItem`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalItem {
    /// The user query to send to the agent.
    pub query: String,

    /// The expected response text, if any, for ground-truth comparison.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_response: Option<String>,

    /// Expected tool calls the agent should make.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expected_tool_calls: Vec<ExpectedToolCall>,

    /// Arbitrary metadata attached to this eval item.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl EvalItem {
    /// Create a new eval item with the given query.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            expected_response: None,
            expected_tool_calls: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the expected response text.
    pub fn with_expected_response(mut self, response: impl Into<String>) -> Self {
        self.expected_response = Some(response.into());
        self
    }

    /// Add an expected tool call.
    pub fn with_expected_tool_call(mut self, tool_call: ExpectedToolCall) -> Self {
        self.expected_tool_calls.push(tool_call);
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

// ---------------------------------------------------------------------------
// Score and result types
// ---------------------------------------------------------------------------

/// The score produced by a single evaluator on a single item.
///
/// Corresponds to Python's `EvalScoreResult`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalScore {
    /// Numeric score (typically 0.0 to 1.0, but evaluator-defined).
    pub score: f64,

    /// Whether the item passed this evaluator's threshold.
    pub passed: bool,

    /// Optional human-readable reason for the score.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl EvalScore {
    /// Create a passing score.
    pub fn pass(score: f64) -> Self {
        Self {
            score,
            passed: true,
            reason: None,
        }
    }

    /// Create a failing score with a reason.
    pub fn fail(score: f64, reason: impl Into<String>) -> Self {
        Self {
            score,
            passed: false,
            reason: Some(reason.into()),
        }
    }
}

/// Per-item result from an evaluation run.
///
/// Contains the original item, the agent's response, and scores from each
/// evaluator. Corresponds to Python's `EvalItemResult`.
#[derive(Debug, Clone)]
pub struct EvalItemResult {
    /// The eval item that was evaluated.
    pub item: EvalItem,

    /// The agent's response.
    pub response: AgentResponse,

    /// Per-evaluator scores, keyed by evaluator name.
    pub scores: HashMap<String, EvalScore>,
}

impl EvalItemResult {
    /// Whether this item passed all evaluators.
    pub fn passed(&self) -> bool {
        self.scores.values().all(|s| s.passed)
    }

    /// Whether this item failed at least one evaluator.
    pub fn failed(&self) -> bool {
        !self.passed()
    }
}

/// Aggregated results from an evaluation run.
///
/// Contains per-item results with convenience methods for computing pass rates
/// and average scores. Corresponds to Python's `EvalResults`.
#[derive(Debug, Clone)]
pub struct EvalResults {
    /// Per-item results.
    pub items: Vec<EvalItemResult>,
}

impl EvalResults {
    /// Create empty results.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// The fraction of items that passed all evaluators (0.0 to 1.0).
    ///
    /// Returns 0.0 if there are no items.
    pub fn pass_rate(&self) -> f64 {
        if self.items.is_empty() {
            return 0.0;
        }
        let passed = self.items.iter().filter(|r| r.passed()).count();
        passed as f64 / self.items.len() as f64
    }

    /// The average score across all items and all evaluators.
    ///
    /// Returns 0.0 if there are no scores.
    pub fn average_score(&self) -> f64 {
        let mut total = 0.0;
        let mut count = 0usize;
        for item_result in &self.items {
            for score in item_result.scores.values() {
                total += score.score;
                count += 1;
            }
        }
        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }

    /// The average score for a specific evaluator across all items.
    ///
    /// Returns `None` if the evaluator was not found in any item.
    pub fn average_score_for(&self, evaluator_name: &str) -> Option<f64> {
        let mut total = 0.0;
        let mut count = 0usize;
        for item_result in &self.items {
            if let Some(score) = item_result.scores.get(evaluator_name) {
                total += score.score;
                count += 1;
            }
        }
        if count == 0 {
            None
        } else {
            Some(total / count as f64)
        }
    }

    /// Whether all items passed all evaluators.
    pub fn all_passed(&self) -> bool {
        !self.items.is_empty() && self.items.iter().all(|r| r.passed())
    }

    /// The total number of evaluated items.
    pub fn total(&self) -> usize {
        self.items.len()
    }

    /// The number of items that passed all evaluators.
    pub fn passed_count(&self) -> usize {
        self.items.iter().filter(|r| r.passed()).count()
    }

    /// The number of items that failed at least one evaluator.
    pub fn failed_count(&self) -> usize {
        self.items.iter().filter(|r| r.failed()).count()
    }
}

impl Default for EvalResults {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Evaluator trait
// ---------------------------------------------------------------------------

/// Trait for evaluation logic that scores an agent's response against an eval item.
///
/// Corresponds to Python's `Evaluator` protocol. Implementations are stateless
/// scorers — the orchestration is handled by [`evaluate_agent`].
#[async_trait]
pub trait Evaluator: Send + Sync {
    /// A unique name for this evaluator (used as the key in score maps).
    fn name(&self) -> &str;

    /// Evaluate an agent's response against the given item.
    async fn evaluate(&self, item: &EvalItem, response: &AgentResponse) -> AgentResult<EvalScore>;
}

// ---------------------------------------------------------------------------
// Built-in evaluators
// ---------------------------------------------------------------------------

/// Evaluator that checks if the agent's response contains expected keywords.
///
/// Returns a score of `(matched / total)` and passes if all keywords are found.
/// Matching is case-insensitive.
pub struct KeywordCheckEvaluator {
    /// Keywords that must appear in the response.
    pub keywords: Vec<String>,
}

impl KeywordCheckEvaluator {
    /// Create a new keyword check evaluator.
    pub fn new(keywords: Vec<String>) -> Self {
        Self { keywords }
    }
}

#[async_trait]
impl Evaluator for KeywordCheckEvaluator {
    fn name(&self) -> &str {
        "keyword_check"
    }

    async fn evaluate(&self, _item: &EvalItem, response: &AgentResponse) -> AgentResult<EvalScore> {
        if self.keywords.is_empty() {
            return Ok(EvalScore::pass(1.0));
        }

        let text_lower = response.text.to_lowercase();
        let matched = self
            .keywords
            .iter()
            .filter(|kw| text_lower.contains(&kw.to_lowercase()))
            .count();

        let score = matched as f64 / self.keywords.len() as f64;
        if matched == self.keywords.len() {
            Ok(EvalScore::pass(score))
        } else {
            let missing: Vec<&str> = self
                .keywords
                .iter()
                .filter(|kw| !text_lower.contains(&kw.to_lowercase()))
                .map(|kw| kw.as_str())
                .collect();
            Ok(EvalScore::fail(
                score,
                format!("missing keywords: {}", missing.join(", ")),
            ))
        }
    }
}

/// Evaluator that checks if the agent called the expected tools.
///
/// Compares the tool calls in the agent's response messages against the
/// `expected_tool_calls` on the [`EvalItem`]. Returns a score of
/// `(matched / expected)` and passes if all expected tools were called.
pub struct ToolCalledEvaluator;

#[async_trait]
impl Evaluator for ToolCalledEvaluator {
    fn name(&self) -> &str {
        "tool_called"
    }

    async fn evaluate(&self, item: &EvalItem, response: &AgentResponse) -> AgentResult<EvalScore> {
        if item.expected_tool_calls.is_empty() {
            return Ok(EvalScore::pass(1.0));
        }

        // Collect all tool call names from the response messages.
        let called_tools: Vec<(&str, &serde_json::Value)> = response
            .messages
            .iter()
            .flat_map(|m| m.tool_calls())
            .map(|(_id, name, args)| (name, args))
            .collect();

        let mut matched = 0usize;
        let mut missing = Vec::new();

        for expected in &item.expected_tool_calls {
            let found = called_tools.iter().any(|(name, args)| {
                if *name != expected.name {
                    return false;
                }
                // If expected args are specified, check them.
                match &expected.args {
                    Some(expected_args) => *args == expected_args,
                    None => true,
                }
            });
            if found {
                matched += 1;
            } else {
                missing.push(expected.name.as_str());
            }
        }

        let score = matched as f64 / item.expected_tool_calls.len() as f64;
        if matched == item.expected_tool_calls.len() {
            Ok(EvalScore::pass(score))
        } else {
            Ok(EvalScore::fail(
                score,
                format!("tools not called: {}", missing.join(", ")),
            ))
        }
    }
}

/// Evaluator that checks the agent's response is non-empty.
///
/// Passes with score 1.0 if the response text is non-empty (after trimming),
/// fails with score 0.0 otherwise.
pub struct ResponseNotEmptyEvaluator;

#[async_trait]
impl Evaluator for ResponseNotEmptyEvaluator {
    fn name(&self) -> &str {
        "response_not_empty"
    }

    async fn evaluate(&self, _item: &EvalItem, response: &AgentResponse) -> AgentResult<EvalScore> {
        if response.text.trim().is_empty() {
            Ok(EvalScore::fail(0.0, "response is empty"))
        } else {
            Ok(EvalScore::pass(1.0))
        }
    }
}

// ---------------------------------------------------------------------------
// ClosureEvaluator
// ---------------------------------------------------------------------------

/// An evaluator built from an async closure.
///
/// Use [`evaluator_fn`] to construct. Mirrors the `tool_fn` pattern from
/// `agent-framework-core`.
pub struct ClosureEvaluator<F> {
    eval_name: String,
    func: F,
}

/// Create an [`Evaluator`] from an async closure.
///
/// # Example
///
/// ```rust
/// use agent_framework_evaluation::{evaluator_fn, EvalItem, EvalScore};
/// use agent_framework_core::types::AgentResponse;
///
/// let evaluator = evaluator_fn("length_check", |_item: &EvalItem, response: &AgentResponse| {
///     let len = response.text.len();
///     async move {
///         if len > 10 {
///             Ok(EvalScore::pass(1.0))
///         } else {
///             Ok(EvalScore::fail(0.0, "response too short"))
///         }
///     }
/// });
/// ```
pub fn evaluator_fn<F, Fut>(name: impl Into<String>, func: F) -> ClosureEvaluator<F>
where
    F: Fn(&EvalItem, &AgentResponse) -> Fut + Send + Sync,
    Fut: Future<Output = AgentResult<EvalScore>> + Send,
{
    ClosureEvaluator {
        eval_name: name.into(),
        func,
    }
}

#[async_trait]
impl<F, Fut> Evaluator for ClosureEvaluator<F>
where
    F: Fn(&EvalItem, &AgentResponse) -> Fut + Send + Sync,
    Fut: Future<Output = AgentResult<EvalScore>> + Send,
{
    fn name(&self) -> &str {
        &self.eval_name
    }

    async fn evaluate(&self, item: &EvalItem, response: &AgentResponse) -> AgentResult<EvalScore> {
        (self.func)(item, response).await
    }
}

// ---------------------------------------------------------------------------
// BoxEvaluator — for type-erased async closures
// ---------------------------------------------------------------------------

/// A type-erased evaluator that wraps a boxed async closure.
///
/// Unlike [`ClosureEvaluator`] (which is generic over `F`), this type uses
/// dynamic dispatch and can be stored directly in a `Vec<Box<dyn Evaluator>>`.
pub struct BoxEvaluator {
    eval_name: String,
    #[allow(clippy::type_complexity)]
    func: Box<
        dyn for<'a> Fn(
                &'a EvalItem,
                &'a AgentResponse,
            ) -> Pin<Box<dyn Future<Output = AgentResult<EvalScore>> + Send + 'a>>
            + Send
            + Sync,
    >,
}

impl BoxEvaluator {
    /// Create a type-erased evaluator from a closure that returns a pinned future.
    pub fn new<F>(name: impl Into<String>, func: F) -> Self
    where
        F: for<'a> Fn(
                &'a EvalItem,
                &'a AgentResponse,
            ) -> Pin<Box<dyn Future<Output = AgentResult<EvalScore>> + Send + 'a>>
            + Send
            + Sync
            + 'static,
    {
        Self {
            eval_name: name.into(),
            func: Box::new(func),
        }
    }
}

#[async_trait]
impl Evaluator for BoxEvaluator {
    fn name(&self) -> &str {
        &self.eval_name
    }

    async fn evaluate(&self, item: &EvalItem, response: &AgentResponse) -> AgentResult<EvalScore> {
        (self.func)(item, response).await
    }
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

/// Run each [`EvalItem`] through the agent, evaluate with all evaluators,
/// and return aggregated [`EvalResults`].
///
/// This is the main entry point for running evaluations. It:
/// 1. Creates a fresh [`AgentSession`] per item (isolated evaluation).
/// 2. Sends the item's query to the agent.
/// 3. Runs all evaluators against the (item, response) pair.
/// 4. Collects results into [`EvalResults`].
///
/// Corresponds to Python's `evaluate_agent` function.
///
/// # Errors
///
/// Returns an error if the agent fails to produce a response for any item.
/// Individual evaluator errors are propagated as-is.
pub async fn evaluate_agent(
    agent: &dyn Agent,
    items: Vec<EvalItem>,
    evaluators: &[Box<dyn Evaluator>],
) -> AgentResult<EvalResults> {
    let mut results = EvalResults::new();

    for item in items {
        let mut session = AgentSession::new();
        let messages = vec![Message::user(&item.query)];
        let response = agent.run(messages, &mut session, None).await?;

        let mut scores = HashMap::new();
        for evaluator in evaluators {
            let score = evaluator.evaluate(&item, &response).await?;
            scores.insert(evaluator.name().to_string(), score);
        }

        results.items.push(EvalItemResult {
            item,
            response,
            scores,
        });
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::types::{AgentResponse, Content, Message, Role};

    /// Helper to build a minimal `AgentResponse` with the given text.
    fn make_response(text: &str) -> AgentResponse {
        AgentResponse {
            messages: vec![Message::assistant(text)],
            text: text.to_string(),
            finish_reason: None,
            usage: None,
        }
    }

    /// Helper to build an `AgentResponse` that includes tool calls.
    fn make_response_with_tools(text: &str, tool_calls: Vec<(&str, &str, serde_json::Value)>) -> AgentResponse {
        let mut contents: Vec<Content> = tool_calls
            .into_iter()
            .map(|(id, name, args)| Content::tool_call(id, name, args))
            .collect();
        contents.push(Content::text(text));

        let msg = Message {
            role: Role::Assistant,
            content: contents,
            name: None,
            metadata: std::collections::HashMap::new(),
        };

        AgentResponse {
            messages: vec![msg],
            text: text.to_string(),
            finish_reason: None,
            usage: None,
        }
    }

    // -- KeywordCheckEvaluator -------------------------------------------------

    #[tokio::test]
    async fn keyword_check_all_present() {
        let evaluator = KeywordCheckEvaluator::new(vec!["sunny".into(), "warm".into()]);
        let item = EvalItem::new("weather?");
        let response = make_response("It is sunny and warm today.");

        let score = evaluator.evaluate(&item, &response).await.unwrap();
        assert!(score.passed);
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn keyword_check_partial_match() {
        let evaluator = KeywordCheckEvaluator::new(vec!["sunny".into(), "cold".into()]);
        let item = EvalItem::new("weather?");
        let response = make_response("It is sunny today.");

        let score = evaluator.evaluate(&item, &response).await.unwrap();
        assert!(!score.passed);
        assert!((score.score - 0.5).abs() < f64::EPSILON);
        assert!(score.reason.as_deref().unwrap().contains("cold"));
    }

    #[tokio::test]
    async fn keyword_check_case_insensitive() {
        let evaluator = KeywordCheckEvaluator::new(vec!["SUNNY".into()]);
        let item = EvalItem::new("weather?");
        let response = make_response("It is sunny today.");

        let score = evaluator.evaluate(&item, &response).await.unwrap();
        assert!(score.passed);
    }

    #[tokio::test]
    async fn keyword_check_empty_keywords() {
        let evaluator = KeywordCheckEvaluator::new(vec![]);
        let item = EvalItem::new("anything");
        let response = make_response("anything");

        let score = evaluator.evaluate(&item, &response).await.unwrap();
        assert!(score.passed);
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    // -- ToolCalledEvaluator ---------------------------------------------------

    #[tokio::test]
    async fn tool_called_all_present() {
        let evaluator = ToolCalledEvaluator;
        let item = EvalItem::new("weather?")
            .with_expected_tool_call(ExpectedToolCall::named("get_weather"));

        let response = make_response_with_tools(
            "The weather is nice.",
            vec![("call_1", "get_weather", serde_json::json!({"city": "Seattle"}))],
        );

        let score = evaluator.evaluate(&item, &response).await.unwrap();
        assert!(score.passed);
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn tool_called_missing() {
        let evaluator = ToolCalledEvaluator;
        let item = EvalItem::new("weather?")
            .with_expected_tool_call(ExpectedToolCall::named("get_weather"))
            .with_expected_tool_call(ExpectedToolCall::named("get_forecast"));

        let response = make_response_with_tools(
            "The weather is nice.",
            vec![("call_1", "get_weather", serde_json::json!({}))],
        );

        let score = evaluator.evaluate(&item, &response).await.unwrap();
        assert!(!score.passed);
        assert!((score.score - 0.5).abs() < f64::EPSILON);
        assert!(score.reason.as_deref().unwrap().contains("get_forecast"));
    }

    #[tokio::test]
    async fn tool_called_with_args_match() {
        let evaluator = ToolCalledEvaluator;
        let expected_args = serde_json::json!({"city": "Seattle"});
        let item = EvalItem::new("weather?")
            .with_expected_tool_call(ExpectedToolCall::with_args("get_weather", expected_args.clone()));

        let response = make_response_with_tools(
            "The weather is nice.",
            vec![("call_1", "get_weather", expected_args)],
        );

        let score = evaluator.evaluate(&item, &response).await.unwrap();
        assert!(score.passed);
    }

    #[tokio::test]
    async fn tool_called_with_args_mismatch() {
        let evaluator = ToolCalledEvaluator;
        let item = EvalItem::new("weather?")
            .with_expected_tool_call(ExpectedToolCall::with_args(
                "get_weather",
                serde_json::json!({"city": "Seattle"}),
            ));

        let response = make_response_with_tools(
            "The weather is nice.",
            vec![("call_1", "get_weather", serde_json::json!({"city": "Portland"}))],
        );

        let score = evaluator.evaluate(&item, &response).await.unwrap();
        assert!(!score.passed);
    }

    #[tokio::test]
    async fn tool_called_no_expected() {
        let evaluator = ToolCalledEvaluator;
        let item = EvalItem::new("hello");
        let response = make_response("Hi there!");

        let score = evaluator.evaluate(&item, &response).await.unwrap();
        assert!(score.passed);
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    // -- ResponseNotEmptyEvaluator ---------------------------------------------

    #[tokio::test]
    async fn response_not_empty_passes() {
        let evaluator = ResponseNotEmptyEvaluator;
        let item = EvalItem::new("hello");
        let response = make_response("Hi there!");

        let score = evaluator.evaluate(&item, &response).await.unwrap();
        assert!(score.passed);
    }

    #[tokio::test]
    async fn response_not_empty_fails_empty() {
        let evaluator = ResponseNotEmptyEvaluator;
        let item = EvalItem::new("hello");
        let response = make_response("");

        let score = evaluator.evaluate(&item, &response).await.unwrap();
        assert!(!score.passed);
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn response_not_empty_fails_whitespace() {
        let evaluator = ResponseNotEmptyEvaluator;
        let item = EvalItem::new("hello");
        let response = make_response("   \n\t  ");

        let score = evaluator.evaluate(&item, &response).await.unwrap();
        assert!(!score.passed);
    }

    // -- ClosureEvaluator (evaluator_fn) ---------------------------------------

    #[tokio::test]
    async fn closure_evaluator_works() {
        let evaluator = evaluator_fn("length_check", |_item: &EvalItem, response: &AgentResponse| {
            let len = response.text.len();
            async move {
                if len > 5 {
                    Ok(EvalScore::pass(1.0))
                } else {
                    Ok(EvalScore::fail(0.0, "too short"))
                }
            }
        });

        assert_eq!(evaluator.name(), "length_check");

        let item = EvalItem::new("test");
        let response = make_response("Hello, world!");
        let score = evaluator.evaluate(&item, &response).await.unwrap();
        assert!(score.passed);

        let short_response = make_response("Hi");
        let score = evaluator.evaluate(&item, &short_response).await.unwrap();
        assert!(!score.passed);
    }

    // -- EvalResults -----------------------------------------------------------

    #[tokio::test]
    async fn eval_results_pass_rate() {
        let mut results = EvalResults::new();
        results.items.push(EvalItemResult {
            item: EvalItem::new("q1"),
            response: make_response("a1"),
            scores: HashMap::from([("e1".into(), EvalScore::pass(1.0))]),
        });
        results.items.push(EvalItemResult {
            item: EvalItem::new("q2"),
            response: make_response("a2"),
            scores: HashMap::from([("e1".into(), EvalScore::fail(0.0, "bad"))]),
        });

        assert!((results.pass_rate() - 0.5).abs() < f64::EPSILON);
        assert!((results.average_score() - 0.5).abs() < f64::EPSILON);
        assert!(!results.all_passed());
        assert_eq!(results.total(), 2);
        assert_eq!(results.passed_count(), 1);
        assert_eq!(results.failed_count(), 1);
    }

    #[tokio::test]
    async fn eval_results_empty() {
        let results = EvalResults::new();
        assert!((results.pass_rate() - 0.0).abs() < f64::EPSILON);
        assert!((results.average_score() - 0.0).abs() < f64::EPSILON);
        assert!(!results.all_passed());
    }

    #[tokio::test]
    async fn eval_results_average_score_for() {
        let mut results = EvalResults::new();
        results.items.push(EvalItemResult {
            item: EvalItem::new("q1"),
            response: make_response("a1"),
            scores: HashMap::from([
                ("e1".into(), EvalScore::pass(0.8)),
                ("e2".into(), EvalScore::pass(1.0)),
            ]),
        });
        results.items.push(EvalItemResult {
            item: EvalItem::new("q2"),
            response: make_response("a2"),
            scores: HashMap::from([
                ("e1".into(), EvalScore::pass(0.6)),
                ("e2".into(), EvalScore::fail(0.2, "low")),
            ]),
        });

        let avg_e1 = results.average_score_for("e1").unwrap();
        assert!((avg_e1 - 0.7).abs() < f64::EPSILON);

        let avg_e2 = results.average_score_for("e2").unwrap();
        assert!((avg_e2 - 0.6).abs() < f64::EPSILON);

        assert!(results.average_score_for("nonexistent").is_none());
    }

    // -- EvalItem builder ------------------------------------------------------

    #[test]
    fn eval_item_builder() {
        let item = EvalItem::new("What is 2+2?")
            .with_expected_response("4")
            .with_expected_tool_call(ExpectedToolCall::named("calculator"))
            .with_metadata("difficulty", serde_json::json!("easy"));

        assert_eq!(item.query, "What is 2+2?");
        assert_eq!(item.expected_response.as_deref(), Some("4"));
        assert_eq!(item.expected_tool_calls.len(), 1);
        assert_eq!(item.expected_tool_calls[0].name, "calculator");
        assert_eq!(item.metadata["difficulty"], serde_json::json!("easy"));
    }

    // -- EvalScore constructors ------------------------------------------------

    #[test]
    fn eval_score_constructors() {
        let pass = EvalScore::pass(0.9);
        assert!(pass.passed);
        assert!((pass.score - 0.9).abs() < f64::EPSILON);
        assert!(pass.reason.is_none());

        let fail = EvalScore::fail(0.1, "low quality");
        assert!(!fail.passed);
        assert!((fail.score - 0.1).abs() < f64::EPSILON);
        assert_eq!(fail.reason.as_deref(), Some("low quality"));
    }
}
