// Copyright (c) Microsoft. All rights reserved.

//! # Hello Agent
//!
//! A minimal example demonstrating a single-turn conversation with an
//! Anthropic-powered agent using the umbrella `agent_framework` crate.
//!
//! ## Prerequisites
//!
//! Set the `ANTHROPIC_API_KEY` environment variable before running:
//!
//! ```sh
//! export ANTHROPIC_API_KEY="sk-ant-..."
//! cargo run -p hello-agent
//! ```

use agent_framework::anthropic::{AnthropicChatClient, AnthropicConfig};
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AnthropicConfig::from_env()?;

    let agent = ChatClientAgent::builder()
        .client(AnthropicChatClient::new(config)?)
        .name("HelloAgent")
        .instructions("You are a friendly assistant. Keep responses brief and cheerful.")
        .build()?;

    let mut session = AgentSession::new();
    let response = agent
        .run(
            vec![Message::user("Tell me a joke about Rust programming.")],
            &mut session,
            None,
        )
        .await?;

    println!("{}", response.text);
    Ok(())
}
