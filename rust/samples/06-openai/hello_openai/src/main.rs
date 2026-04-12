// Copyright (c) Microsoft. All rights reserved.

//! # Hello OpenAI
//!
//! Minimal example using the OpenAI provider. Mirrors `01-get-started/hello_agent`
//! but targets the OpenAI Chat Completions API instead of Anthropic.
//!
//! ## Prerequisites
//!
//! ```sh
//! export OPENAI_API_KEY="sk-..."
//! cargo run -p hello-openai
//! ```

use agent_framework_core::agent::{Agent, ChatClientAgent};
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::Message;
use agent_framework_openai::{OpenAIChatClient, OpenAIConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = OpenAIConfig::from_env()?;

    let agent = ChatClientAgent::builder()
        .client(OpenAIChatClient::new(config)?)
        .name("OpenAIHelloAgent")
        .instructions("You are a helpful assistant. Keep responses concise.")
        .build()?;

    let mut session = AgentSession::new();
    let response = agent
        .run(
            vec![Message::user("What are the key differences between Rust and Go?")],
            &mut session,
            None,
        )
        .await?;

    println!("{}", response.text);
    Ok(())
}
