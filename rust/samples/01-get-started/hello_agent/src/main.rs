// Copyright (c) Microsoft. All rights reserved.

//! # Hello Agent
//!
//! A minimal example demonstrating a single-turn conversation with an
//! Anthropic-powered agent.
//!
//! ## Prerequisites
//!
//! Set the `ANTHROPIC_API_KEY` environment variable before running:
//!
//! ```sh
//! export ANTHROPIC_API_KEY="sk-ant-..."
//! cargo run -p hello-agent
//! ```

use agent_framework_anthropic::{AnthropicChatClient, AnthropicConfig};
use agent_framework_core::agent::{Agent, ChatClientAgent};
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration from environment variables.
    let config = AnthropicConfig::from_env()?;

    // Build an agent backed by the Anthropic client.
    let agent = ChatClientAgent::builder()
        .client(AnthropicChatClient::new(config))
        .name("HelloAgent")
        .instructions("You are a friendly assistant. Keep responses brief and cheerful.")
        .build()?;

    // Create a session and run.
    let mut session = AgentSession::new();
    let response = agent
        .run(
            vec![Message::user("Tell me a joke about Rust programming.")],
            &mut session,
        )
        .await?;

    println!("{}", response.text);

    Ok(())
}
