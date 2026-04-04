# Agent Framework for Rust

The Rust SDK for the [Microsoft Agent Framework](../README.md) — build, orchestrate, and deploy AI agents.

## Crate Structure

| Crate | Description |
|-------|-------------|
| `agent-framework-core` | Core abstractions: Agent, ChatClient, Tools, Middleware, Session |
| `agent-framework-anthropic` | Anthropic Claude provider |
| `agent-framework-openai` | OpenAI provider |
| `agent-framework` | Umbrella crate that re-exports everything |

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
agent-framework = "0.1"
tokio = { version = "1", features = ["full"] }
```

### Hello Agent

```rust
use agent_framework::prelude::*;
use agent_framework::anthropic::{AnthropicChatClient, AnthropicConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AnthropicConfig::from_env()?;
    let agent = ChatClientAgent::builder()
        .client(AnthropicChatClient::new(config))
        .instructions("You are a helpful assistant.")
        .build()?;

    let mut session = AgentSession::new();
    let response = agent
        .run(vec![Message::user("Hello!")], &mut session)
        .await?;

    println!("{}", response.text);
    Ok(())
}
```

## Architecture

The Rust SDK mirrors the abstractions found in the Python and .NET SDKs:

- **`Agent` trait** — Core interface for running conversations (`run`, `run_stream`)
- **`ChatClientAgent`** — Primary implementation backed by a `ChatClient` (LLM provider)
- **`ChatClient` trait** — Provider abstraction (Anthropic, OpenAI, etc.)
- **`FunctionTool` trait** — Tools the agent can invoke during conversations
- **Middleware** — Three-layer pipeline (agent, chat-client, function levels)
- **`AgentSession`** — Conversation state and history management

## Development

```sh
# Build all crates
cargo build --all

# Run tests
cargo test --all

# Check formatting
cargo fmt --all --check

# Run clippy
cargo clippy --all-targets --all-features -- -D warnings
```

## Samples

See [`samples/`](./samples/) for runnable examples:

- [`01-get-started/hello_agent`](./samples/01-get-started/hello_agent/) — Minimal single-turn agent

## Requirements

- Rust 1.75+ (2021 edition)
- Tokio async runtime
