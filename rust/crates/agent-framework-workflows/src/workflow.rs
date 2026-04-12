// Copyright (c) Microsoft. All rights reserved.

//! Workflow execution engine.

use std::collections::{HashMap, HashSet};

use tracing::{debug, instrument};

use agent_framework_core::error::{AgentError, AgentResult};

use crate::edge::{Edge, EdgeTarget};
use crate::executor::Executor;
use crate::state::{CheckpointStorage, WorkflowCheckpoint, WorkflowState};

/// A compiled workflow DAG ready for execution.
///
/// Mirrors .NET's `WorkflowHostAgent` and Python's `Workflow`.
pub struct Workflow {
    nodes: HashMap<String, Box<dyn Executor>>,
    edges: Vec<Edge>,
    entry_node: String,
    /// Optional checkpoint storage for suspend/resume.
    checkpoint_storage: Option<Box<dyn CheckpointStorage>>,
}

/// Events emitted during workflow execution (for streaming / observability).
#[derive(Debug, Clone)]
pub enum WorkflowEvent {
    /// A node started executing.
    NodeStarted { node_id: String },
    /// A node completed successfully.
    NodeCompleted { node_id: String },
    /// A node failed.
    NodeFailed { node_id: String, error: String },
    /// The workflow completed.
    WorkflowCompleted,
    /// A checkpoint was saved.
    CheckpointSaved { run_id: String },
}

/// The result of a workflow run.
#[derive(Debug, Clone)]
pub struct WorkflowRunResult {
    /// Final state after all nodes have executed.
    pub state: HashMap<String, serde_json::Value>,
    /// Events emitted during the run.
    pub events: Vec<WorkflowEvent>,
}

impl Workflow {
    /// Run the workflow with a fresh state.
    pub async fn run(&self) -> AgentResult<WorkflowRunResult> {
        let state = WorkflowState::new();
        self.run_with_state(&state).await?;
        Ok(WorkflowRunResult {
            state: state.snapshot().await,
            events: Vec::new(), // TODO: capture events via channel
        })
    }

    /// Run the workflow with the given initial state.
    pub async fn run_with_initial_state(
        &self,
        initial: HashMap<String, serde_json::Value>,
    ) -> AgentResult<WorkflowRunResult> {
        let state = WorkflowState::from_map(initial);
        self.run_with_state(&state).await?;
        Ok(WorkflowRunResult {
            state: state.snapshot().await,
            events: Vec::new(),
        })
    }

    /// Run the workflow sharing the given state object.
    #[instrument(skip(self, state))]
    pub async fn run_with_state(&self, state: &WorkflowState) -> AgentResult<()> {
        let run_id = uuid::Uuid::new_v4().to_string();
        let mut completed: HashSet<String> = HashSet::new();
        let mut pending: Vec<String> = vec![self.entry_node.clone()];

        // Resume from checkpoint if available.
        if let Some(storage) = &self.checkpoint_storage {
            if let Some(cp) = storage.load(&run_id).await? {
                state.restore(cp.state).await;
                completed = cp.completed_nodes.into_iter().collect();
                pending = cp.pending_nodes;
            }
        }

        let max_steps = self.nodes.len() * 10; // safety limit
        let mut step = 0;

        while !pending.is_empty() && step < max_steps {
            step += 1;
            let current_batch = std::mem::take(&mut pending);
            let mut next_pending = Vec::new();

            // Execute nodes in the current batch.
            // If there are multiple (fan-out), run them concurrently.
            if current_batch.len() == 1 {
                let node_id = &current_batch[0];
                self.execute_node(node_id, state, &mut completed).await?;
                self.resolve_next(node_id, state, &completed, &mut next_pending)
                    .await?;
            } else {
                // Concurrent execution for fan-out.
                for node_id in &current_batch {
                    let node = self.nodes.get(node_id).ok_or_else(|| {
                        AgentError::InvalidRequest(format!("unknown node: {node_id}"))
                    })?;
                    let state_clone = state.clone();
                    let id = node_id.clone();
                    // We can't move the executor into a task (it's behind &dyn),
                    // so we execute sequentially for now.
                    // TODO: Support true parallel execution with Arc<dyn Executor>.
                    debug!(node_id = %id, "Executing fan-out node");
                    node.execute(&state_clone).await.map_err(|e| {
                        AgentError::InvalidResponse(format!("node '{id}' failed: {e}"))
                    })?;
                    completed.insert(id);
                }

                // Resolve next for all completed nodes.
                for node_id in &current_batch {
                    self.resolve_next(node_id, state, &completed, &mut next_pending)
                        .await?;
                }
            }

            // Save checkpoint after each step if storage is configured.
            if let Some(storage) = &self.checkpoint_storage {
                let cp = WorkflowCheckpoint {
                    run_id: run_id.clone(),
                    state: state.snapshot().await,
                    completed_nodes: completed.iter().cloned().collect(),
                    pending_nodes: next_pending.clone(),
                };
                storage.save(&cp).await?;
            }

            pending = next_pending;
        }

        if step >= max_steps && !pending.is_empty() {
            return Err(AgentError::InvalidResponse(format!(
                "workflow exceeded maximum steps ({max_steps}); possible cycle"
            )));
        }

        Ok(())
    }

    async fn execute_node(
        &self,
        node_id: &str,
        state: &WorkflowState,
        completed: &mut HashSet<String>,
    ) -> AgentResult<()> {
        let node = self.nodes.get(node_id).ok_or_else(|| {
            AgentError::InvalidRequest(format!("unknown node: {node_id}"))
        })?;

        debug!(node_id, "Executing node");
        node.execute(state).await.map_err(|e| {
            AgentError::InvalidResponse(format!("node '{node_id}' failed: {e}"))
        })?;
        completed.insert(node_id.to_string());
        debug!(node_id, "Node completed");
        Ok(())
    }

    async fn resolve_next(
        &self,
        node_id: &str,
        state: &WorkflowState,
        completed: &HashSet<String>,
        pending: &mut Vec<String>,
    ) -> AgentResult<()> {
        for edge in &self.edges {
            let is_source = edge.from_node().map(|f| f == node_id).unwrap_or(false);
            if !is_source {
                // Check fan-in edges: if all wait_for nodes are done, add target.
                if let Edge::FanIn { wait_for, to } = edge {
                    if wait_for.iter().all(|n| completed.contains(n))
                        && !completed.contains(to)
                        && !pending.contains(to)
                    {
                        pending.push(to.clone());
                    }
                }
                continue;
            }

            let target = edge.resolve(state).await;
            match target {
                EdgeTarget::Single(t) if !completed.contains(&t) => {
                    if !pending.contains(&t) {
                        pending.push(t);
                    }
                }
                EdgeTarget::FanOut(targets) => {
                    for t in targets {
                        if !completed.contains(&t) && !pending.contains(&t) {
                            pending.push(t);
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// WorkflowBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing a [`Workflow`].
pub struct WorkflowBuilder {
    nodes: HashMap<String, Box<dyn Executor>>,
    edges: Vec<Edge>,
    entry_node: Option<String>,
    checkpoint_storage: Option<Box<dyn CheckpointStorage>>,
}

impl WorkflowBuilder {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            entry_node: None,
            checkpoint_storage: None,
        }
    }

    /// Add a node to the workflow.
    pub fn add_node(mut self, id: impl Into<String>, executor: impl Executor + 'static) -> Self {
        let id = id.into();
        self.nodes.insert(id, Box::new(executor));
        self
    }

    /// Add a direct (unconditional) edge.
    pub fn add_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.push(Edge::direct(from, to));
        self
    }

    /// Add a conditional edge.
    pub fn add_conditional_edge<F>(mut self, from: impl Into<String>, condition: F) -> Self
    where
        F: Fn(&WorkflowState) -> std::pin::Pin<Box<dyn std::future::Future<Output = EdgeTarget> + Send + '_>>
            + Send
            + Sync
            + 'static,
    {
        self.edges.push(Edge::Conditional {
            from: from.into(),
            condition: Box::new(condition),
        });
        self
    }

    /// Add a fan-out edge (parallel execution).
    pub fn add_fan_out(mut self, from: impl Into<String>, targets: Vec<String>) -> Self {
        self.edges.push(Edge::fan_out(from, targets));
        self
    }

    /// Add a fan-in edge (join).
    pub fn add_fan_in(mut self, wait_for: Vec<String>, to: impl Into<String>) -> Self {
        self.edges.push(Edge::fan_in(wait_for, to));
        self
    }

    /// Add a switch/case edge.
    pub fn add_switch(
        mut self,
        from: impl Into<String>,
        key: impl Into<String>,
        cases: Vec<(serde_json::Value, String)>,
        default: Option<String>,
    ) -> Self {
        self.edges.push(Edge::switch_case(from, key, cases, default));
        self
    }

    /// Add a raw edge.
    pub fn add_raw_edge(mut self, edge: Edge) -> Self {
        self.edges.push(edge);
        self
    }

    /// Set the entry node (the first node to execute).
    pub fn set_entry(mut self, node_id: impl Into<String>) -> Self {
        self.entry_node = Some(node_id.into());
        self
    }

    /// Enable checkpoint storage for suspend/resume.
    pub fn with_checkpoint_storage(mut self, storage: impl CheckpointStorage + 'static) -> Self {
        self.checkpoint_storage = Some(Box::new(storage));
        self
    }

    /// Build the workflow.
    pub fn build(self) -> AgentResult<Workflow> {
        let entry = self
            .entry_node
            .or_else(|| self.nodes.keys().next().cloned())
            .ok_or_else(|| AgentError::InvalidRequest("workflow has no nodes".to_string()))?;

        if !self.nodes.contains_key(&entry) {
            return Err(AgentError::InvalidRequest(format!(
                "entry node '{entry}' not found in workflow"
            )));
        }

        Ok(Workflow {
            nodes: self.nodes,
            edges: self.edges,
            entry_node: entry,
            checkpoint_storage: self.checkpoint_storage,
        })
    }
}

impl Default for WorkflowBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::FunctionExecutor;

    #[tokio::test]
    async fn simple_linear_workflow() {
        let workflow = WorkflowBuilder::new()
            .add_node(
                "step1",
                FunctionExecutor::new(|state| {
                    Box::pin(async move {
                        state.set("value", serde_json::json!(1)).await;
                        Ok(())
                    })
                })
                .with_id("step1"),
            )
            .add_node(
                "step2",
                FunctionExecutor::new(|state| {
                    Box::pin(async move {
                        let v: i64 = state.get_as("value").await.unwrap_or(0);
                        state.set("value", serde_json::json!(v + 10)).await;
                        Ok(())
                    })
                })
                .with_id("step2"),
            )
            .add_edge("step1", "step2")
            .set_entry("step1")
            .build()
            .unwrap();

        let result = workflow.run().await.unwrap();
        assert_eq!(result.state.get("value").unwrap(), &serde_json::json!(11));
    }

    #[tokio::test]
    async fn switch_case_routing() {
        let workflow = WorkflowBuilder::new()
            .add_node(
                "start",
                FunctionExecutor::new(|state| {
                    Box::pin(async move {
                        state.set("route", serde_json::json!("b")).await;
                        Ok(())
                    })
                })
                .with_id("start"),
            )
            .add_node(
                "path_a",
                FunctionExecutor::new(|state| {
                    Box::pin(async move {
                        state.set("result", serde_json::json!("took A")).await;
                        Ok(())
                    })
                })
                .with_id("path_a"),
            )
            .add_node(
                "path_b",
                FunctionExecutor::new(|state| {
                    Box::pin(async move {
                        state.set("result", serde_json::json!("took B")).await;
                        Ok(())
                    })
                })
                .with_id("path_b"),
            )
            .add_switch(
                "start",
                "route",
                vec![
                    (serde_json::json!("a"), "path_a".to_string()),
                    (serde_json::json!("b"), "path_b".to_string()),
                ],
                None,
            )
            .set_entry("start")
            .build()
            .unwrap();

        let result = workflow.run().await.unwrap();
        assert_eq!(result.state.get("result").unwrap(), &serde_json::json!("took B"));
    }

    #[tokio::test]
    async fn fan_out_fan_in() {
        let workflow = WorkflowBuilder::new()
            .add_node(
                "start",
                FunctionExecutor::new(|state| {
                    Box::pin(async move {
                        state.set("started", serde_json::json!(true)).await;
                        Ok(())
                    })
                })
                .with_id("start"),
            )
            .add_node(
                "branch_a",
                FunctionExecutor::new(|state| {
                    Box::pin(async move {
                        state.set("a_done", serde_json::json!(true)).await;
                        Ok(())
                    })
                })
                .with_id("branch_a"),
            )
            .add_node(
                "branch_b",
                FunctionExecutor::new(|state| {
                    Box::pin(async move {
                        state.set("b_done", serde_json::json!(true)).await;
                        Ok(())
                    })
                })
                .with_id("branch_b"),
            )
            .add_node(
                "join",
                FunctionExecutor::new(|state| {
                    Box::pin(async move {
                        let a = state.get_as::<bool>("a_done").await.unwrap_or(false);
                        let b = state.get_as::<bool>("b_done").await.unwrap_or(false);
                        state.set("both_done", serde_json::json!(a && b)).await;
                        Ok(())
                    })
                })
                .with_id("join"),
            )
            .add_fan_out("start", vec!["branch_a".to_string(), "branch_b".to_string()])
            .add_fan_in(vec!["branch_a".to_string(), "branch_b".to_string()], "join")
            .set_entry("start")
            .build()
            .unwrap();

        let result = workflow.run().await.unwrap();
        assert_eq!(result.state.get("both_done").unwrap(), &serde_json::json!(true));
    }

    #[tokio::test]
    async fn workflow_with_initial_state() {
        let workflow = WorkflowBuilder::new()
            .add_node(
                "double",
                FunctionExecutor::new(|state| {
                    Box::pin(async move {
                        let v: i64 = state.get_as("x").await.unwrap_or(0);
                        state.set("result", serde_json::json!(v * 2)).await;
                        Ok(())
                    })
                })
                .with_id("double"),
            )
            .set_entry("double")
            .build()
            .unwrap();

        let mut initial = HashMap::new();
        initial.insert("x".to_string(), serde_json::json!(21));
        let result = workflow.run_with_initial_state(initial).await.unwrap();
        assert_eq!(result.state.get("result").unwrap(), &serde_json::json!(42));
    }
}
