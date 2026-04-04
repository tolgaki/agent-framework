// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework Core
//!
//! Core abstractions for the Microsoft Agent Framework.
//!
//! This crate provides the foundational types and traits for building AI agents:
//!
//! - [`agent::Agent`] — The core agent trait for running conversations.
//! - [`agent::ChatClientAgent`] — The primary agent implementation backed by an LLM.
//! - [`client::ChatClient`] — The provider abstraction for LLM communication.
//! - [`tools::FunctionTool`] — Tools that agents can invoke during conversations.
//! - [`middleware`] — Three-layer middleware pipeline (agent, client, function).
//! - [`session::AgentSession`] — Session and conversation state management.
//! - [`types`] — Unified message and content types.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use agent_framework_core::agent::{Agent, ChatClientAgent};
//! use agent_framework_core::session::AgentSession;
//! use agent_framework_core::types::Message;
//!
//! # async fn example(client: Box<dyn agent_framework_core::client::ChatClient>) {
//! let agent = ChatClientAgent::builder()
//!     .client_boxed(client)
//!     .instructions("You are a helpful assistant.")
//!     .build()
//!     .unwrap();
//!
//! let mut session = AgentSession::new();
//! let response = agent.run(vec![Message::user("Hello!")], &mut session).await.unwrap();
//! println!("{}", response.text);
//! # }
//! ```

pub mod agent;
pub mod client;
pub mod context;
pub mod error;
pub mod middleware;
pub mod session;
pub mod streaming;
pub mod tools;
pub mod types;

// Re-export commonly used items at the crate root.
pub use agent::{Agent, ChatClientAgent};
pub use client::ChatClient;
pub use error::{AgentError, AgentResult};
pub use session::AgentSession;
pub use tools::{FunctionTool, ToolDefinition};
pub use types::{AgentResponse, ChatOptions, ChatResponse, Content, Message, Role};
