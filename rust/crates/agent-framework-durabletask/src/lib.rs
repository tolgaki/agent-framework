// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework — Durable task execution
//!
//! A durable execution wrapper for long-running agents. Mirrors the .NET
//! `Microsoft.Agents.AI.DurableTask` project in spirit: callers schedule an
//! agent run, receive a task id immediately, and later poll for the status
//! and final result. State transitions are persisted through a [`TaskStore`]
//! so execution can be observed (and, with a durable store implementation,
//! survive process restarts).
//!
//! # State machine
//!
//! ```text
//! Pending -> Running -> Completed
//!                    -> Failed
//!                    -> Cancelled
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use std::sync::Arc;
//!
//! use agent_framework_core::agent::Agent;
//! use agent_framework_core::types::Message;
//! use agent_framework_durabletask::{DurableAgent, InMemoryTaskStore, TaskStore};
//!
//! # async fn example(inner: Arc<dyn Agent>) -> agent_framework_core::error::AgentResult<()> {
//! let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
//! let durable = DurableAgent::new(inner, store);
//!
//! let task_id = durable.start_run(vec![Message::user("Plan a trip")], None).await?;
//! // ... later ...
//! if let Some(response) = durable.get_result(&task_id).await? {
//!     println!("{}", response.text);
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::debug;

use agent_framework_core::agent::Agent;
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::{AgentResponse, AgentRunOptions, Message};

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

/// The lifecycle state of a [`DurableTask`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DurableTaskState {
    /// The task has been created but the background worker has not yet picked it up.
    Pending,
    /// The background worker is actively executing the agent.
    Running,
    /// The agent finished successfully. See [`DurableTask::result`] for the response.
    Completed,
    /// The agent returned an error. See [`DurableTask::error`] for the message.
    Failed,
    /// The task was cancelled before it completed.
    Cancelled,
}

impl DurableTaskState {
    /// Returns `true` if the task is in a terminal state.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    /// Returns `true` if the task has not yet reached a terminal state.
    pub fn is_active(self) -> bool {
        !self.is_terminal()
    }
}

/// A persisted durable task record.
///
/// `T` is typically [`AgentResponse`] but is generic so callers can plug in
/// their own result types when wrapping non-agent work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurableTask<T> {
    /// Stable task identifier (UUID v4 as a string).
    pub id: String,

    /// Current lifecycle state.
    pub state: DurableTaskState,

    /// The successful result, if the task has completed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<T>,

    /// The error message, if the task failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    /// ISO-8601 timestamp when the task was created.
    pub created_at: String,

    /// ISO-8601 timestamp of the last state transition.
    pub updated_at: String,
}

impl<T> DurableTask<T> {
    /// Create a new task in the [`DurableTaskState::Pending`] state with a
    /// fresh UUID v4 identifier.
    pub fn new() -> Self {
        let now = current_timestamp();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            state: DurableTaskState::Pending,
            result: None,
            error: None,
            created_at: now.clone(),
            updated_at: now,
        }
    }

    /// Create a new task with the given identifier.
    pub fn with_id(id: impl Into<String>) -> Self {
        let now = current_timestamp();
        Self {
            id: id.into(),
            state: DurableTaskState::Pending,
            result: None,
            error: None,
            created_at: now.clone(),
            updated_at: now,
        }
    }

    /// Transition to a new state and bump `updated_at`.
    pub fn transition(&mut self, state: DurableTaskState) {
        self.state = state;
        self.updated_at = current_timestamp();
    }
}

impl<T> Default for DurableTask<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Produce a coarse ISO-8601 timestamp (to-the-second, UTC) using only the
/// standard library. Callers that want sub-second precision or a different
/// format should replace this helper — the string is opaque to the store.
fn current_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = now.as_secs();
    let days = secs / 86_400;
    let rem = secs % 86_400;
    let hour = rem / 3600;
    let minute = (rem % 3600) / 60;
    let second = rem % 60;
    // Civil-from-days algorithm (Howard Hinnant, public domain).
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", year, m, d, hour, minute, second)
}

// ---------------------------------------------------------------------------
// TaskStore
// ---------------------------------------------------------------------------

/// An async store for durable task state.
///
/// Values are stored as opaque JSON so callers can evolve the schema without
/// breaking older records.
#[async_trait]
pub trait TaskStore: Send + Sync {
    /// Persist `state` under `task_id`, overwriting any existing record.
    async fn save(&self, task_id: &str, state: serde_json::Value) -> AgentResult<()>;

    /// Load the state for `task_id`, or `None` if it does not exist.
    async fn load(&self, task_id: &str) -> AgentResult<Option<serde_json::Value>>;

    /// Delete the record for `task_id`. Deleting a missing task is a no-op.
    async fn delete(&self, task_id: &str) -> AgentResult<()>;
}

/// An in-memory [`TaskStore`] suitable for tests and single-process use.
///
/// Data is lost when the process exits. Production deployments should plug in
/// a durable store (Redis, Cosmos DB, Postgres, etc.).
#[derive(Debug, Clone, Default)]
pub struct InMemoryTaskStore {
    inner: Arc<RwLock<HashMap<String, serde_json::Value>>>,
}

impl InMemoryTaskStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current number of persisted tasks (for tests / debugging).
    pub async fn len(&self) -> usize {
        self.inner.read().await.len()
    }

    /// Whether the store currently holds no tasks.
    pub async fn is_empty(&self) -> bool {
        self.inner.read().await.is_empty()
    }
}

#[async_trait]
impl TaskStore for InMemoryTaskStore {
    async fn save(&self, task_id: &str, state: serde_json::Value) -> AgentResult<()> {
        let mut map = self.inner.write().await;
        map.insert(task_id.to_string(), state);
        Ok(())
    }

    async fn load(&self, task_id: &str) -> AgentResult<Option<serde_json::Value>> {
        let map = self.inner.read().await;
        Ok(map.get(task_id).cloned())
    }

    async fn delete(&self, task_id: &str) -> AgentResult<()> {
        let mut map = self.inner.write().await;
        map.remove(task_id);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DurableAgent
// ---------------------------------------------------------------------------

/// A [`DurableTask`] specialized on [`AgentResponse`].
pub type AgentDurableTask = DurableTask<AgentResponse>;

/// Wraps an [`Agent`] with durable execution semantics.
///
/// Callers hand work to the agent via [`start_run`](Self::start_run), which
/// returns a task id immediately. The agent is executed on a background
/// tokio task; its lifecycle transitions (Pending -> Running -> Completed /
/// Failed) are persisted to the configured [`TaskStore`] so the caller can
/// poll at any time via [`get_status`](Self::get_status) and
/// [`get_result`](Self::get_result).
///
/// Note that the spawned worker executes the agent against a **fresh**
/// [`AgentSession`] per run, because a mutable session reference cannot be
/// moved across the `'static` boundary required by `tokio::spawn`. Callers
/// that need to thread session state across durable runs should configure a
/// persistent history provider on the wrapped agent and restore it inside
/// the worker.
pub struct DurableAgent {
    inner: Arc<dyn Agent>,
    store: Arc<dyn TaskStore>,
}

impl DurableAgent {
    /// Create a new durable wrapper around `inner`, persisting state to `store`.
    pub fn new(inner: Arc<dyn Agent>, store: Arc<dyn TaskStore>) -> Self {
        Self { inner, store }
    }

    /// Return the wrapped agent.
    pub fn inner(&self) -> &Arc<dyn Agent> {
        &self.inner
    }

    /// Return the configured task store.
    pub fn store(&self) -> &Arc<dyn TaskStore> {
        &self.store
    }

    /// Schedule an agent run and return the new task id.
    ///
    /// The run executes on a background tokio task. The task is initially
    /// recorded as [`DurableTaskState::Pending`] and transitions to
    /// [`Running`](DurableTaskState::Running) once the worker starts, then to
    /// [`Completed`](DurableTaskState::Completed) or
    /// [`Failed`](DurableTaskState::Failed) when the agent finishes.
    pub async fn start_run(
        &self,
        messages: Vec<Message>,
        options: Option<AgentRunOptions>,
    ) -> AgentResult<String> {
        let task: AgentDurableTask = DurableTask::new();
        let task_id = task.id.clone();

        // Persist the initial Pending record before handing off so callers
        // that poll immediately see a non-missing task.
        let initial = serde_json::to_value(&task)?;
        self.store.save(&task_id, initial).await?;

        let agent = self.inner.clone();
        let store = self.store.clone();
        let worker_task_id = task_id.clone();

        tokio::spawn(async move {
            if let Err(e) = run_worker(agent, store, worker_task_id.clone(), messages, options).await {
                // run_worker already attempts to persist a Failed record on
                // inner errors; this only fires if the store itself breaks.
                debug!(task_id = %worker_task_id, error = %e, "DurableAgent: worker loop crashed");
            }
        });

        Ok(task_id)
    }

    /// Fetch the current state of a task. Returns an error if the task does
    /// not exist in the store.
    pub async fn get_status(&self, task_id: &str) -> AgentResult<DurableTaskState> {
        let task = self.load_task(task_id).await?;
        Ok(task.state)
    }

    /// Fetch the final result of a task.
    ///
    /// - `Ok(Some(response))` — the task completed successfully.
    /// - `Ok(None)` — the task exists but is not yet complete (Pending / Running).
    /// - `Err(AgentError::ProviderError)` — the task failed; the message contains the error.
    /// - `Err(AgentError::InvalidRequest)` — the task was cancelled or does not exist.
    pub async fn get_result(&self, task_id: &str) -> AgentResult<Option<AgentResponse>> {
        let task = self.load_task(task_id).await?;
        match task.state {
            DurableTaskState::Completed => Ok(task.result),
            DurableTaskState::Failed => Err(AgentError::provider(
                task.error.unwrap_or_else(|| "durable task failed".to_string()),
                None,
            )),
            DurableTaskState::Cancelled => Err(AgentError::InvalidRequest(format!(
                "durable task {task_id} was cancelled"
            ))),
            DurableTaskState::Pending | DurableTaskState::Running => Ok(None),
        }
    }

    /// Load and parse a task record from the store.
    async fn load_task(&self, task_id: &str) -> AgentResult<AgentDurableTask> {
        let raw = self
            .store
            .load(task_id)
            .await?
            .ok_or_else(|| AgentError::InvalidRequest(format!("durable task {task_id} not found")))?;
        let task: AgentDurableTask = serde_json::from_value(raw)?;
        Ok(task)
    }
}

/// The background worker that actually runs the agent and persists state
/// transitions. Extracted out of [`DurableAgent::start_run`] so it can be
/// tested independently and so errors from the inner run do not escape the
/// spawned future.
async fn run_worker(
    agent: Arc<dyn Agent>,
    store: Arc<dyn TaskStore>,
    task_id: String,
    messages: Vec<Message>,
    options: Option<AgentRunOptions>,
) -> AgentResult<()> {
    // Move to Running.
    let mut task: AgentDurableTask = match store.load(&task_id).await? {
        Some(v) => serde_json::from_value(v)?,
        None => {
            debug!(task_id = %task_id, "DurableAgent: worker started but record missing; creating");
            DurableTask::with_id(&task_id)
        }
    };
    task.transition(DurableTaskState::Running);
    store.save(&task_id, serde_json::to_value(&task)?).await?;
    debug!(task_id = %task_id, "DurableAgent: transitioned to Running");

    // Execute the agent. We cannot carry a borrowed session across `spawn`,
    // so each durable run gets a fresh in-memory session. Callers that need
    // persistent history should configure a history provider on the agent.
    let mut session = AgentSession::new();
    let result = agent.run(messages, &mut session, options.as_ref()).await;

    match result {
        Ok(response) => {
            task.result = Some(response);
            task.transition(DurableTaskState::Completed);
            debug!(task_id = %task_id, "DurableAgent: completed");
        }
        Err(err) => {
            task.error = Some(err.to_string());
            task.transition(DurableTaskState::Failed);
            debug!(task_id = %task_id, "DurableAgent: failed");
        }
    }

    store.save(&task_id, serde_json::to_value(&task)?).await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use agent_framework_core::streaming::AgentResponseStream;
    use agent_framework_core::types::{AgentResponse, Message};

    // ----- Test doubles -----

    /// An agent that always succeeds, echoing back the concatenated input text.
    struct EchoAgent {
        id: String,
    }

    #[async_trait]
    impl Agent for EchoAgent {
        fn id(&self) -> &str {
            &self.id
        }
        fn name(&self) -> Option<&str> {
            Some("echo")
        }
        fn description(&self) -> Option<&str> {
            None
        }
        async fn run(
            &self,
            messages: Vec<Message>,
            _session: &mut AgentSession,
            _options: Option<&AgentRunOptions>,
        ) -> AgentResult<AgentResponse> {
            let text = messages.iter().map(|m| m.text()).collect::<Vec<_>>().join(" | ");
            Ok(AgentResponse {
                messages: vec![Message::assistant(text.clone())],
                text,
                finish_reason: None,
                usage: None,
            })
        }
        fn run_stream<'a>(
            &'a self,
            _messages: Vec<Message>,
            _session: &'a mut AgentSession,
            _options: Option<&'a AgentRunOptions>,
        ) -> AgentResult<AgentResponseStream<'a>> {
            Err(AgentError::Unimplemented("EchoAgent does not stream".to_string()))
        }
    }

    /// An agent that always fails with a provider error.
    struct FailingAgent;

    #[async_trait]
    impl Agent for FailingAgent {
        fn id(&self) -> &str {
            "failing"
        }
        fn name(&self) -> Option<&str> {
            None
        }
        fn description(&self) -> Option<&str> {
            None
        }
        async fn run(
            &self,
            _messages: Vec<Message>,
            _session: &mut AgentSession,
            _options: Option<&AgentRunOptions>,
        ) -> AgentResult<AgentResponse> {
            Err(AgentError::provider("boom", Some(500)))
        }
        fn run_stream<'a>(
            &'a self,
            _messages: Vec<Message>,
            _session: &'a mut AgentSession,
            _options: Option<&'a AgentRunOptions>,
        ) -> AgentResult<AgentResponseStream<'a>> {
            Err(AgentError::Unimplemented("FailingAgent does not stream".to_string()))
        }
    }

    // ----- Helpers -----

    /// Spin until the task reaches a terminal state or we exhaust our budget.
    async fn wait_terminal(agent: &DurableAgent, id: &str) -> DurableTaskState {
        use std::time::Duration;
        for _ in 0..200 {
            let state = agent.get_status(id).await.unwrap();
            if state.is_terminal() {
                return state;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("durable task {id} did not reach a terminal state within budget");
    }

    // ----- Tests -----

    #[test]
    fn state_helpers_classify_terminal_vs_active() {
        assert!(DurableTaskState::Pending.is_active());
        assert!(DurableTaskState::Running.is_active());
        assert!(!DurableTaskState::Completed.is_active());
        assert!(DurableTaskState::Completed.is_terminal());
        assert!(DurableTaskState::Failed.is_terminal());
        assert!(DurableTaskState::Cancelled.is_terminal());
        assert!(!DurableTaskState::Pending.is_terminal());
    }

    #[test]
    fn transition_updates_state_and_timestamp() {
        let mut task: AgentDurableTask = DurableTask::new();
        assert_eq!(task.state, DurableTaskState::Pending);
        let created = task.updated_at.clone();
        // Timestamps are second-granularity, so force a distinct value.
        std::thread::sleep(std::time::Duration::from_millis(1100));
        task.transition(DurableTaskState::Running);
        assert_eq!(task.state, DurableTaskState::Running);
        assert_ne!(task.updated_at, created, "updated_at should advance after transition");
    }

    #[tokio::test]
    async fn in_memory_store_roundtrip() {
        let store = InMemoryTaskStore::new();
        assert!(store.is_empty().await);

        let task: AgentDurableTask = DurableTask::with_id("t1");
        let value = serde_json::to_value(&task).unwrap();
        store.save("t1", value.clone()).await.unwrap();
        assert_eq!(store.len().await, 1);

        let loaded = store.load("t1").await.unwrap().expect("must exist");
        let parsed: AgentDurableTask = serde_json::from_value(loaded).unwrap();
        assert_eq!(parsed.id, "t1");
        assert_eq!(parsed.state, DurableTaskState::Pending);

        store.delete("t1").await.unwrap();
        assert!(store.load("t1").await.unwrap().is_none());
        assert!(store.is_empty().await);
    }

    #[tokio::test]
    async fn save_load_preserves_result_and_error_fields() {
        let store = InMemoryTaskStore::new();
        let mut task: AgentDurableTask = DurableTask::with_id("t2");
        task.result = Some(AgentResponse {
            messages: vec![Message::assistant("hi")],
            text: "hi".into(),
            finish_reason: None,
            usage: None,
        });
        task.transition(DurableTaskState::Completed);

        store.save("t2", serde_json::to_value(&task).unwrap()).await.unwrap();
        let loaded: AgentDurableTask =
            serde_json::from_value(store.load("t2").await.unwrap().unwrap()).unwrap();
        assert_eq!(loaded.state, DurableTaskState::Completed);
        assert_eq!(loaded.result.as_ref().map(|r| r.text.as_str()), Some("hi"));
        assert!(loaded.error.is_none());
    }

    #[tokio::test]
    async fn agent_run_completes_successfully() {
        let agent: Arc<dyn Agent> = Arc::new(EchoAgent { id: "echo-1".into() });
        let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
        let durable = DurableAgent::new(agent, store);

        let id = durable
            .start_run(vec![Message::user("hello"), Message::user("world")], None)
            .await
            .unwrap();

        let state = wait_terminal(&durable, &id).await;
        assert_eq!(state, DurableTaskState::Completed);

        let response = durable
            .get_result(&id)
            .await
            .unwrap()
            .expect("completed task must carry a result");
        assert_eq!(response.text, "hello | world");
    }

    #[tokio::test]
    async fn agent_run_failure_is_persisted() {
        let agent: Arc<dyn Agent> = Arc::new(FailingAgent);
        let store: Arc<dyn TaskStore> = Arc::new(InMemoryTaskStore::new());
        let durable = DurableAgent::new(agent, store);

        let id = durable.start_run(vec![Message::user("go")], None).await.unwrap();
        let state = wait_terminal(&durable, &id).await;
        assert_eq!(state, DurableTaskState::Failed);

        let err = durable
            .get_result(&id)
            .await
            .expect_err("failed task must surface an error");
        assert!(format!("{err}").contains("boom"), "error message should contain inner failure");
    }
}
