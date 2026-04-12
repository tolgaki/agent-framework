// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework Declarative Workflows
//!
//! Declarative JSON/YAML workflow definitions for the Microsoft Agent
//! Framework. Mirrors .NET's `Microsoft.Agents.AI.Workflows.Declarative`.
//!
//! A [`WorkflowDefinition`] is a serde-friendly description of a DAG. It is
//! turned into an executable [`Workflow`] by a [`WorkflowFactory`], which
//! owns registries of executor factories for named node types.
//!
//! # Example
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use agent_framework_workflows::{FunctionExecutor, WorkflowState};
//! use agent_framework_workflows_declarative::{WorkflowDefinition, WorkflowFactory};
//!
//! let definition = WorkflowDefinition::from_json(r#"{
//!     "name": "demo",
//!     "entry": "greet",
//!     "nodes": [
//!         { "id": "greet", "type": "function", "config": { "name": "greet_user" } }
//!     ],
//!     "edges": []
//! }"#).unwrap();
//!
//! let mut factory = WorkflowFactory::new();
//! factory.register_function("greet_user", |id: String, _config| {
//!     Box::new(
//!         FunctionExecutor::new(|state: &WorkflowState| {
//!             Box::pin(async move {
//!                 state.set("greeted", serde_json::json!(true)).await;
//!                 Ok(())
//!             })
//!         })
//!         .with_id(id),
//!     )
//! });
//!
//! let workflow = factory.build(&definition).unwrap();
//! # let _ = workflow;
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::debug;

use agent_framework_core::agent::Agent;
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_workflows::{AgentExecutor, Edge, Executor, Workflow, WorkflowBuilder};

// ---------------------------------------------------------------------------
// Definition types
// ---------------------------------------------------------------------------

/// The top-level declarative workflow definition.
///
/// Mirrors .NET's `WorkflowDefinition` and Python's declarative YAML schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDefinition {
    /// Human-readable workflow name.
    pub name: String,

    /// Optional description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// The ID of the entry node.
    pub entry: String,

    /// The set of nodes in the workflow.
    #[serde(default)]
    pub nodes: Vec<NodeDefinition>,

    /// The set of edges in the workflow.
    #[serde(default)]
    pub edges: Vec<EdgeDefinition>,
}

/// A node in a declarative workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDefinition {
    /// The node's unique ID within the workflow.
    pub id: String,

    /// The node type. One of `"function"`, `"agent"`, or `"workflow"`.
    #[serde(rename = "type")]
    pub kind: String,

    /// Executor-specific configuration.
    #[serde(default)]
    pub config: serde_json::Value,
}

/// An edge in a declarative workflow.
///
/// Uses the [internally-tagged][serde-tag] serde representation so that JSON
/// such as `{"type": "direct", "from": "a", "to": "b"}` round-trips cleanly.
///
/// [serde-tag]: https://serde.rs/enum-representations.html#internally-tagged
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EdgeDefinition {
    /// Unconditional edge from `from` to `to`.
    Direct { from: String, to: String },

    /// Fan-out edge: from one node to multiple targets in parallel.
    FanOut {
        from: String,
        targets: Vec<String>,
    },

    /// Fan-in edge: wait for all `wait_for` nodes before running `to`.
    FanIn {
        wait_for: Vec<String>,
        to: String,
    },

    /// Switch/case edge: dispatch on a state key.
    SwitchCase {
        from: String,
        key: String,
        cases: Vec<(serde_json::Value, String)>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        default: Option<String>,
    },
}

impl WorkflowDefinition {
    /// Parse a definition from a JSON string.
    pub fn from_json(json: &str) -> AgentResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| AgentError::InvalidRequest(format!("invalid workflow definition: {e}")))
    }

    /// Load a definition from a JSON file on disk.
    ///
    /// Uses synchronous filesystem I/O. Workflow definitions are typically
    /// loaded once at startup, so this does not justify pulling in tokio's
    /// `fs` feature.
    pub fn from_file(path: impl AsRef<Path>) -> AgentResult<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| {
            AgentError::InvalidRequest(format!(
                "failed to read workflow definition '{}': {e}",
                path.display()
            ))
        })?;
        let text = std::str::from_utf8(&bytes).map_err(|e| {
            AgentError::InvalidRequest(format!(
                "workflow definition '{}' is not valid UTF-8: {e}",
                path.display()
            ))
        })?;
        Self::from_json(text)
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// A factory that turns a [`NodeDefinition`] of type `"function"` into a
/// boxed [`Executor`].
///
/// The factory receives the node's ID (so the executor can be identified in
/// the DAG) and the node's `config` blob.
pub type FunctionExecutorFactory =
    Arc<dyn Fn(String, &serde_json::Value) -> Box<dyn Executor> + Send + Sync>;

/// A factory that turns a [`NodeDefinition`] of type `"agent"` into a shared
/// [`Agent`]. The factory is consulted by name when building a workflow.
pub type AgentFactory =
    Arc<dyn Fn(&serde_json::Value) -> AgentResult<Arc<dyn Agent>> + Send + Sync>;

/// Registry-backed builder that turns a [`WorkflowDefinition`] into a runnable
/// [`Workflow`].
///
/// Users register factories for the named function executors and agents their
/// workflows reference. The node `config` values are threaded through to those
/// factories verbatim.
pub struct WorkflowFactory {
    functions: HashMap<String, FunctionExecutorFactory>,
    agents: HashMap<String, AgentFactory>,
}

impl WorkflowFactory {
    /// Create an empty factory.
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            agents: HashMap::new(),
        }
    }

    /// Register a factory for a `function`-type node.
    ///
    /// The `name` is matched against the `name` field of the node's `config`
    /// object (or, for back-compat, against the node ID if `name` is absent).
    pub fn register_function<F>(&mut self, name: impl Into<String>, factory: F) -> &mut Self
    where
        F: Fn(String, &serde_json::Value) -> Box<dyn Executor> + Send + Sync + 'static,
    {
        self.functions.insert(name.into(), Arc::new(factory));
        self
    }

    /// Register a factory for an `agent`-type node.
    ///
    /// The `name` is matched against the `name` field of the node's `config`
    /// object (or, for back-compat, against the node ID if `name` is absent).
    pub fn register_agent<F>(&mut self, name: impl Into<String>, factory: F) -> &mut Self
    where
        F: Fn(&serde_json::Value) -> AgentResult<Arc<dyn Agent>> + Send + Sync + 'static,
    {
        self.agents.insert(name.into(), Arc::new(factory));
        self
    }

    /// Build a [`Workflow`] from the given definition.
    pub fn build(&self, definition: &WorkflowDefinition) -> AgentResult<Workflow> {
        debug!(workflow = %definition.name, "Building declarative workflow");

        let mut builder = WorkflowBuilder::new().set_entry(definition.entry.clone());

        for node in &definition.nodes {
            let executor = self.build_node(node)?;
            builder = builder.add_node(node.id.clone(), BoxedExecutor(executor));
        }

        for edge in &definition.edges {
            builder = builder.add_raw_edge(edge_to_runtime(edge));
        }

        builder.build()
    }

    fn build_node(&self, node: &NodeDefinition) -> AgentResult<Box<dyn Executor>> {
        let name = node
            .config
            .get("name")
            .and_then(|v| v.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| node.id.clone());

        match node.kind.as_str() {
            "function" => {
                let factory = self.functions.get(&name).ok_or_else(|| {
                    AgentError::InvalidRequest(format!(
                        "no function factory registered for '{name}' (node '{}')",
                        node.id
                    ))
                })?;
                Ok(factory(node.id.clone(), &node.config))
            }
            "agent" => {
                let factory = self.agents.get(&name).ok_or_else(|| {
                    AgentError::InvalidRequest(format!(
                        "no agent factory registered for '{name}' (node '{}')",
                        node.id
                    ))
                })?;
                let agent = factory(&node.config)?;

                let mut executor = AgentExecutor::new(agent).with_id(node.id.clone());
                if let Some(input) = node.config.get("input_key").and_then(|v| v.as_str()) {
                    executor = executor.with_input_key(input);
                }
                if let Some(output) = node.config.get("output_key").and_then(|v| v.as_str()) {
                    executor = executor.with_output_key(output);
                }
                if let Some(response) = node.config.get("response_key").and_then(|v| v.as_str()) {
                    executor = executor.with_response_key(response);
                }
                Ok(Box::new(executor))
            }
            "workflow" => {
                // Nested sub-workflows would be recursively built from an
                // inner definition. Declarative composition of sub-workflows
                // is deliberately deferred until we know the user-facing
                // shape the .NET/Python SDKs settle on.
                Err(AgentError::Unimplemented(format!(
                    "declarative sub-workflows are not yet supported (node '{}')",
                    node.id
                )))
            }
            other => Err(AgentError::InvalidRequest(format!(
                "unknown node type '{other}' for node '{}'",
                node.id
            ))),
        }
    }
}

impl Default for WorkflowFactory {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn edge_to_runtime(edge: &EdgeDefinition) -> Edge {
    match edge {
        EdgeDefinition::Direct { from, to } => Edge::direct(from.clone(), to.clone()),
        EdgeDefinition::FanOut { from, targets } => {
            Edge::fan_out(from.clone(), targets.clone())
        }
        EdgeDefinition::FanIn { wait_for, to } => Edge::fan_in(wait_for.clone(), to.clone()),
        EdgeDefinition::SwitchCase {
            from,
            key,
            cases,
            default,
        } => Edge::switch_case(from.clone(), key.clone(), cases.clone(), default.clone()),
    }
}

/// Wraps a `Box<dyn Executor>` so it can be passed to
/// [`WorkflowBuilder::add_node`], which expects an owned `impl Executor`.
struct BoxedExecutor(Box<dyn Executor>);

#[async_trait::async_trait]
impl Executor for BoxedExecutor {
    fn id(&self) -> &str {
        self.0.id()
    }

    async fn execute(
        &self,
        state: &agent_framework_workflows::WorkflowState,
    ) -> AgentResult<()> {
        self.0.execute(state).await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_workflows::{FunctionExecutor, WorkflowState};

    fn noop_function_factory() -> FunctionExecutor {
        FunctionExecutor::new(|state: &WorkflowState| {
            Box::pin(async move {
                state.set("ran", serde_json::json!(true)).await;
                Ok(())
            })
        })
    }

    #[test]
    fn parses_minimal_definition() {
        let json = r#"{
            "name": "demo",
            "entry": "a",
            "nodes": [
                { "id": "a", "type": "function", "config": {} }
            ],
            "edges": []
        }"#;

        let def = WorkflowDefinition::from_json(json).unwrap();
        assert_eq!(def.name, "demo");
        assert_eq!(def.entry, "a");
        assert_eq!(def.nodes.len(), 1);
        assert_eq!(def.nodes[0].kind, "function");
        assert!(def.edges.is_empty());
    }

    #[test]
    fn parses_all_edge_variants() {
        let json = r#"{
            "name": "all-edges",
            "entry": "start",
            "nodes": [],
            "edges": [
                { "type": "direct", "from": "a", "to": "b" },
                { "type": "fan_out", "from": "b", "targets": ["c", "d"] },
                { "type": "fan_in", "wait_for": ["c", "d"], "to": "e" },
                {
                    "type": "switch_case",
                    "from": "e",
                    "key": "route",
                    "cases": [["x", "f"], ["y", "g"]],
                    "default": "h"
                }
            ]
        }"#;

        let def = WorkflowDefinition::from_json(json).unwrap();
        assert_eq!(def.edges.len(), 4);

        match &def.edges[0] {
            EdgeDefinition::Direct { from, to } => {
                assert_eq!(from, "a");
                assert_eq!(to, "b");
            }
            other => panic!("expected direct edge, got {other:?}"),
        }

        match &def.edges[1] {
            EdgeDefinition::FanOut { from, targets } => {
                assert_eq!(from, "b");
                assert_eq!(targets, &vec!["c".to_string(), "d".to_string()]);
            }
            other => panic!("expected fan_out edge, got {other:?}"),
        }

        match &def.edges[2] {
            EdgeDefinition::FanIn { wait_for, to } => {
                assert_eq!(wait_for, &vec!["c".to_string(), "d".to_string()]);
                assert_eq!(to, "e");
            }
            other => panic!("expected fan_in edge, got {other:?}"),
        }

        match &def.edges[3] {
            EdgeDefinition::SwitchCase {
                from,
                key,
                cases,
                default,
            } => {
                assert_eq!(from, "e");
                assert_eq!(key, "route");
                assert_eq!(cases.len(), 2);
                assert_eq!(cases[0].0, serde_json::json!("x"));
                assert_eq!(cases[0].1, "f");
                assert_eq!(default.as_deref(), Some("h"));
            }
            other => panic!("expected switch_case edge, got {other:?}"),
        }
    }

    #[test]
    fn edge_variants_roundtrip() {
        let edges = vec![
            EdgeDefinition::Direct {
                from: "a".into(),
                to: "b".into(),
            },
            EdgeDefinition::FanOut {
                from: "a".into(),
                targets: vec!["b".into(), "c".into()],
            },
            EdgeDefinition::FanIn {
                wait_for: vec!["b".into(), "c".into()],
                to: "d".into(),
            },
            EdgeDefinition::SwitchCase {
                from: "d".into(),
                key: "k".into(),
                cases: vec![(serde_json::json!(1), "e".into())],
                default: None,
            },
        ];

        for edge in edges {
            let json = serde_json::to_string(&edge).unwrap();
            let back: EdgeDefinition = serde_json::from_str(&json).unwrap();
            // Round-trip via JSON should preserve structure.
            let json2 = serde_json::to_string(&back).unwrap();
            assert_eq!(json, json2);
        }
    }

    #[tokio::test]
    async fn builds_workflow_from_minimal_definition() {
        let def = WorkflowDefinition {
            name: "demo".into(),
            description: None,
            entry: "noop".into(),
            nodes: vec![NodeDefinition {
                id: "noop".into(),
                kind: "function".into(),
                config: serde_json::json!({ "name": "noop" }),
            }],
            edges: vec![],
        };

        let mut factory = WorkflowFactory::new();
        factory.register_function("noop", |id, _config| {
            Box::new(noop_function_factory().with_id(id))
        });

        let workflow = factory.build(&def).unwrap();
        let result = workflow.run().await.unwrap();
        assert_eq!(result.state.get("ran"), Some(&serde_json::json!(true)));
    }

    #[tokio::test]
    async fn builds_workflow_with_direct_edge() {
        let def = WorkflowDefinition {
            name: "two-step".into(),
            description: Some("a → b".into()),
            entry: "a".into(),
            nodes: vec![
                NodeDefinition {
                    id: "a".into(),
                    kind: "function".into(),
                    config: serde_json::json!({ "name": "step" }),
                },
                NodeDefinition {
                    id: "b".into(),
                    kind: "function".into(),
                    config: serde_json::json!({ "name": "step" }),
                },
            ],
            edges: vec![EdgeDefinition::Direct {
                from: "a".into(),
                to: "b".into(),
            }],
        };

        let mut factory = WorkflowFactory::new();
        factory.register_function("step", |id, _config| {
            let counter_key = format!("visited_{id}");
            Box::new(
                FunctionExecutor::new(move |state: &WorkflowState| {
                    let key = counter_key.clone();
                    Box::pin(async move {
                        state.set(key, serde_json::json!(true)).await;
                        Ok(())
                    })
                })
                .with_id(id),
            )
        });

        let workflow = factory.build(&def).unwrap();
        let result = workflow.run().await.unwrap();
        assert_eq!(result.state.get("visited_a"), Some(&serde_json::json!(true)));
        assert_eq!(result.state.get("visited_b"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn unknown_node_type_is_rejected() {
        let def = WorkflowDefinition {
            name: "bad".into(),
            description: None,
            entry: "n".into(),
            nodes: vec![NodeDefinition {
                id: "n".into(),
                kind: "alien".into(),
                config: serde_json::Value::Null,
            }],
            edges: vec![],
        };

        let factory = WorkflowFactory::new();
        let result = factory.build(&def);
        match result {
            Err(AgentError::InvalidRequest(msg)) => assert!(msg.contains("alien")),
            Err(other) => panic!("expected InvalidRequest, got {other:?}"),
            Ok(_) => panic!("expected build to fail"),
        }
    }

    #[test]
    fn missing_function_factory_is_rejected() {
        let def = WorkflowDefinition {
            name: "bad".into(),
            description: None,
            entry: "n".into(),
            nodes: vec![NodeDefinition {
                id: "n".into(),
                kind: "function".into(),
                config: serde_json::json!({ "name": "missing" }),
            }],
            edges: vec![],
        };

        let factory = WorkflowFactory::new();
        let result = factory.build(&def);
        match result {
            Err(AgentError::InvalidRequest(msg)) => assert!(msg.contains("missing")),
            Err(other) => panic!("expected InvalidRequest, got {other:?}"),
            Ok(_) => panic!("expected build to fail"),
        }
    }
}
