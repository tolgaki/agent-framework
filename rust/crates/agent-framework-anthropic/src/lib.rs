// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework Anthropic
//!
//! Anthropic Claude provider for the Microsoft Agent Framework.
//!
//! This crate provides [`AnthropicChatClient`], a [`ChatClient`](agent_framework_core::ChatClient)
//! implementation that communicates with the Anthropic Messages API.
//!
//! # Example
//!
//! ```rust,no_run
//! use agent_framework_core::agent::{Agent, ChatClientAgent};
//! use agent_framework_core::session::AgentSession;
//! use agent_framework_core::types::Message;
//! use agent_framework_anthropic::{AnthropicChatClient, AnthropicConfig};
//!
//! # async fn example() -> agent_framework_core::error::AgentResult<()> {
//! let config = AnthropicConfig::from_env()?;
//! let agent = ChatClientAgent::builder()
//!     .client(AnthropicChatClient::new(config)?)
//!     .instructions("You are a helpful assistant.")
//!     .build()?;
//!
//! let mut session = AgentSession::new();
//! let response = agent.run(vec![Message::user("Hello!")], &mut session, None).await?;
//! println!("{}", response.text);
//! # Ok(())
//! # }
//! ```

mod chat_client;

pub use chat_client::{AnthropicChatClient, AnthropicConfig};
