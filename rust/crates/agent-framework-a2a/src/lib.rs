// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework A2A
//!
//! Agent-to-Agent (A2A) protocol integration for the Microsoft Agent Framework,
//! built on top of [`a2a-rs-server`](https://crates.io/crates/a2a-rs-server) and
//! [`a2a-rs-client`](https://crates.io/crates/a2a-rs-client) v1.0.17 (A2A RC 1.0).
//!
//! This crate provides two integrations:
//!
//! 1. **[`A2AAgentHandler`]** — implements `a2a-rs-server::MessageHandler` so any
//!    [`Agent`](agent_framework_core::agent::Agent) can be exposed as an A2A server.
//!    Plug it into [`a2a_rs_server::A2aServer`] to serve agents over the A2A
//!    JSON-RPC protocol.
//!
//! 2. **[`A2ARemoteAgent`]** — implements [`Agent`] using `a2a-rs-client::A2aClient`
//!    so any A2A-compliant remote endpoint can be consumed as if it were a local
//!    agent. Drop it into the framework's middleware, orchestrations, workflows,
//!    or `ChatClientAgent` exactly like any other agent.
//!
//! # Server example
//!
//! ```rust,no_run
//! use std::sync::Arc;
//! use a2a_rs_server::A2aServer;
//! use agent_framework_a2a::A2AAgentHandler;
//! # async fn run(agent: Arc<dyn agent_framework_core::agent::Agent>) -> anyhow::Result<()> {
//! let handler = A2AAgentHandler::new(agent)
//!     .with_organization("Contoso");
//! A2aServer::new(handler).bind("0.0.0.0:8080")?.run().await
//! # }
//! ```
//!
//! # Client example
//!
//! ```rust,no_run
//! use agent_framework_a2a::A2ARemoteAgent;
//! use agent_framework_core::agent::Agent;
//! use agent_framework_core::session::AgentSession;
//! use agent_framework_core::types::Message;
//!
//! # async fn run() -> anyhow::Result<()> {
//! let remote = A2ARemoteAgent::connect("http://localhost:8080").await?;
//! let mut session = AgentSession::new();
//! let response = remote
//!     .run(vec![Message::user("Hello!")], &mut session, None)
//!     .await?;
//! println!("{}", response.text);
//! # Ok(())
//! # }
//! ```

mod handler;
mod remote;

pub use handler::A2AAgentHandler;
pub use remote::{A2AClientConfig, A2ARemoteAgent};

// Re-export the underlying crates for convenience.
pub use a2a_rs_client;
pub use a2a_rs_core;
pub use a2a_rs_server;
