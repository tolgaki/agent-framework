// Copyright (c) Microsoft. All rights reserved.

//! # Multi-turn Conversation
//!
//! Demonstrates how `AgentSession` preserves conversation history across
//! multiple calls to `agent.run()`, so the model can reference earlier turns.
//!
//! ## Prerequisites
//!
//! ```sh
//! export ANTHROPIC_API_KEY="sk-ant-..."
//! cargo run -p conversation
//! ```

use agent_framework_anthropic::{AnthropicChatClient, AnthropicConfig};
use agent_framework_core::agent::{Agent, ChatClientAgent};
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AnthropicConfig::from_env()?;

    let agent = ChatClientAgent::builder()
        .client(AnthropicChatClient::new(config)?)
        .name("ConversationAgent")
        .instructions("You are a helpful assistant. Keep track of what the user tells you.")
        .build()?;

    // A single session carries the conversation history.
    let mut session = AgentSession::new();

    // Turn 1
    let r1 = agent
        .run(
            vec![Message::user("My name is Alice and I like hiking.")],
            &mut session,
            None,
        )
        .await?;
    println!("Assistant: {}\n", r1.text);

    // Turn 2 — the model should remember the user's name.
    let r2 = agent
        .run(
            vec![Message::user("What's my name and what do I like?")],
            &mut session,
            None,
        )
        .await?;
    println!("Assistant: {}\n", r2.text);

    // Turn 3 — demonstrate per-call instruction override.
    let r3 = agent
        .run(
            vec![Message::user("Summarize our conversation so far.")],
            &mut session,
            None,
        )
        .await?;
    println!("Assistant: {}", r3.text);

    Ok(())
}
