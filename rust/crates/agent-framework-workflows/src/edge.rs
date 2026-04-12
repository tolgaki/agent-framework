// Copyright (c) Microsoft. All rights reserved.

//! Workflow edge types for routing between nodes.

use std::future::Future;
use std::pin::Pin;

use crate::state::WorkflowState;

/// Type alias for async edge condition functions.
type ConditionFn = Box<dyn Fn(&WorkflowState) -> Pin<Box<dyn Future<Output = EdgeTarget> + Send + '_>> + Send + Sync>;

/// A routing decision from an edge.
#[derive(Debug, Clone)]
pub enum EdgeTarget {
    /// Route to a single node.
    Single(String),
    /// Route to multiple nodes in parallel (fan-out).
    FanOut(Vec<String>),
    /// No target — the workflow ends after this node.
    End,
}

/// An edge connecting two nodes in the workflow DAG.
pub enum Edge {
    /// Unconditional edge: always route to the target.
    Direct {
        from: String,
        to: String,
    },

    /// Conditional edge: route based on a predicate evaluated against state.
    Conditional {
        from: String,
        condition: ConditionFn,
    },

    /// Fan-out: route to multiple targets in parallel.
    FanOut {
        from: String,
        targets: Vec<String>,
    },

    /// Fan-in: wait for all specified predecessors before proceeding.
    FanIn {
        /// Nodes that must all complete before the target runs.
        wait_for: Vec<String>,
        /// The node to run after all predecessors complete.
        to: String,
    },

    /// Switch/case: evaluate a key in state and route accordingly.
    SwitchCase {
        from: String,
        /// State key to read the switch value from.
        key: String,
        /// Map of case values → target node IDs.
        cases: Vec<(serde_json::Value, String)>,
        /// Default target if no case matches.
        default: Option<String>,
    },
}

impl Edge {
    /// Create an unconditional edge.
    pub fn direct(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self::Direct {
            from: from.into(),
            to: to.into(),
        }
    }

    /// Create a fan-out edge.
    pub fn fan_out(from: impl Into<String>, targets: Vec<String>) -> Self {
        Self::FanOut {
            from: from.into(),
            targets,
        }
    }

    /// Create a fan-in edge.
    pub fn fan_in(wait_for: Vec<String>, to: impl Into<String>) -> Self {
        Self::FanIn {
            wait_for,
            to: to.into(),
        }
    }

    /// Create a switch/case edge.
    pub fn switch_case(
        from: impl Into<String>,
        key: impl Into<String>,
        cases: Vec<(serde_json::Value, String)>,
        default: Option<String>,
    ) -> Self {
        Self::SwitchCase {
            from: from.into(),
            key: key.into(),
            cases,
            default,
        }
    }

    /// Get the source node ID.
    pub fn from_node(&self) -> Option<&str> {
        match self {
            Edge::Direct { from, .. } => Some(from),
            Edge::Conditional { from, .. } => Some(from),
            Edge::FanOut { from, .. } => Some(from),
            Edge::FanIn { .. } => None,
            Edge::SwitchCase { from, .. } => Some(from),
        }
    }

    /// Resolve the next target(s) given the current workflow state.
    pub async fn resolve(&self, state: &WorkflowState) -> EdgeTarget {
        match self {
            Edge::Direct { to, .. } => EdgeTarget::Single(to.clone()),
            Edge::Conditional { condition, .. } => (condition)(state).await,
            Edge::FanOut { targets, .. } => EdgeTarget::FanOut(targets.clone()),
            Edge::FanIn { to, .. } => EdgeTarget::Single(to.clone()),
            Edge::SwitchCase {
                key,
                cases,
                default,
                ..
            } => {
                let value = state.get(key).await;
                for (case_val, target) in cases {
                    if value.as_ref() == Some(case_val) {
                        return EdgeTarget::Single(target.clone());
                    }
                }
                match default {
                    Some(d) => EdgeTarget::Single(d.clone()),
                    None => EdgeTarget::End,
                }
            }
        }
    }
}
