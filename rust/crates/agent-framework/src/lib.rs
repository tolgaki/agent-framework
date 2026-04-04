// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework
//!
//! The Microsoft Agent Framework for Rust.
//!
//! This is the umbrella crate that re-exports the core framework and all
//! enabled provider crates. For most users, adding `agent-framework` to your
//! `Cargo.toml` is all you need.
//!
//! # Features
//!
//! - `anthropic` (default) — Anthropic Claude provider
//! - `openai` (default) — OpenAI provider
//!
//! # Example
//!
//! ```rust,no_run
//! use agent_framework::prelude::*;
//!
//! # async fn example() -> AgentResult<()> {
//! // Use with Anthropic
//! # #[cfg(feature = "anthropic")]
//! let agent = ChatClientAgent::builder()
//!     .client(agent_framework::anthropic::AnthropicChatClient::new(
//!         agent_framework::anthropic::AnthropicConfig::from_env()?,
//!     ))
//!     .instructions("You are a helpful assistant.")
//!     .build()?;
//!
//! let mut session = AgentSession::new();
//! # #[cfg(feature = "anthropic")]
//! let response = agent.run(vec![Message::user("Hello!")], &mut session).await?;
//! # #[cfg(feature = "anthropic")]
//! println!("{}", response.text);
//! # Ok(())
//! # }
//! ```

// Re-export core.
pub use agent_framework_core::*;

/// Anthropic Claude provider.
#[cfg(feature = "anthropic")]
pub mod anthropic {
    pub use agent_framework_anthropic::*;
}

/// OpenAI provider.
#[cfg(feature = "openai")]
pub mod openai {
    pub use agent_framework_openai::*;
}

/// Common imports for quick starts.
pub mod prelude {
    pub use agent_framework_core::agent::{Agent, ChatClientAgent};
    pub use agent_framework_core::client::ChatClient;
    pub use agent_framework_core::error::{AgentError, AgentResult};
    pub use agent_framework_core::session::AgentSession;
    pub use agent_framework_core::tools::{FunctionTool, ToolDefinition};
    pub use agent_framework_core::types::{AgentResponse, ChatOptions, Content, Message, Role};
}
