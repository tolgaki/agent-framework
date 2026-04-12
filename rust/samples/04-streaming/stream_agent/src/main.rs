// Copyright (c) Microsoft. All rights reserved.

//! # Streaming Agent
//!
//! Demonstrates `agent.run_stream()` which yields text tokens as they arrive
//! from the model. Great for real-time UIs and CLI tools that want to show
//! incremental output.
//!
//! ## Prerequisites
//!
//! ```sh
//! export ANTHROPIC_API_KEY="sk-ant-..."
//! cargo run -p stream-agent
//! ```

use std::io::Write;

use agent_framework_anthropic::{AnthropicChatClient, AnthropicConfig};
use agent_framework_core::agent::{Agent, ChatClientAgent};
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::Message;
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AnthropicConfig::from_env()?;

    let agent = ChatClientAgent::builder()
        .client(AnthropicChatClient::new(config)?)
        .name("StreamAgent")
        .instructions("You are a creative storyteller. Write vivid, engaging prose.")
        .build()?;

    let mut session = AgentSession::new();

    println!("--- Streaming response ---\n");

    let stream = agent.run_stream(
        vec![Message::user(
            "Tell me a short story about a Rust programmer who discovers a magical crate.",
        )],
        &mut session,
        None,
    )?;

    // Pin the stream and consume updates as they arrive.
    let mut stream = std::pin::pin!(stream);
    while let Some(update) = stream.next().await {
        let update = update?;
        if let Some(text) = update.text {
            print!("{text}");
            std::io::stdout().flush().ok();
        }
    }
    println!("\n\n--- Stream complete ---");

    Ok(())
}
