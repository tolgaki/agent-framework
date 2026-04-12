// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework Workflows
//!
//! A DAG-based workflow engine for composing agents, functions, and
//! sub-workflows into multi-step pipelines with conditional routing,
//! fan-out/fan-in parallelism, and checkpoint/resume support.
//!
//! Mirrors .NET's `Microsoft.Agents.AI.Workflows` and Python's
//! `agent_framework._workflows` modules.
//!
//! # Architecture
//!
//! - **Nodes** are units of work: functions ([`FunctionExecutor`]), agents
//!   ([`AgentExecutor`]), or nested workflows ([`WorkflowExecutor`]).
//! - **Edges** connect nodes and control routing: unconditional, conditional,
//!   fan-out (parallel), fan-in (join), and switch/case.
//! - The [`WorkflowBuilder`] provides a fluent API for constructing workflows.
//! - The [`Workflow`] runs the DAG, streaming events as nodes complete.
//!
//! # Example
//!
//! ```rust,ignore
//! use agent_framework_workflows::{WorkflowBuilder, FunctionExecutor};
//!
//! let workflow = WorkflowBuilder::new()
//!     .add_node("greet", FunctionExecutor::new(|state| Box::pin(async move {
//!         state.set("greeting", serde_json::json!("Hello!")).await;
//!         Ok(())
//!     })).with_id("greet"))
//!     .add_edge("greet", "respond")
//!     .set_entry("greet")
//!     .build()
//!     .unwrap();
//! ```

mod edge;
mod executor;
mod state;
mod workflow;

pub use edge::*;
pub use executor::*;
pub use state::*;
pub use workflow::*;
