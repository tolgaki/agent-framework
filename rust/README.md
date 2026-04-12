# Agent Framework for Rust

The Rust SDK for the [Microsoft Agent Framework](../README.md) — build, orchestrate, and deploy AI agents.

## Features

- **Agent abstraction** — `ChatClientAgent` wraps any LLM provider with a built-in tool-calling loop, streaming support, and session management.
- **Provider ecosystem** — first-party `AnthropicChatClient` and `OpenAIChatClient`; bring your own by implementing the `ChatClient` trait.
- **Tool calling** — register functions as `FunctionTool` (trait or closure), and the agent handles the model-call / tool-execute / model-continue loop automatically.
- **Three-layer middleware** — intercept at the agent level, chat-client level, or individual tool invocation level. Matches Python's middleware and .NET's `DelegatingAIAgent` / `DelegatingChatClient` patterns.
- **Streaming** — full SSE streaming for both providers with tool-call delta accumulation, plus agent-side `run_stream` that drives the tool loop while yielding incremental text.
- **Per-call options** — override temperature, model, penalties, tool choice, and more per invocation via `AgentRunOptions`, without modifying the agent's defaults.
- **Security-first** — API keys wrapped in `SecretString` (zeroized on drop, redacted in Debug), HTTP redirects disabled, response body size limits, error body scrubbing.

## Crate structure

| Crate | Description |
|-------|-------------|
| [`agent-framework-core`](./crates/agent-framework-core/) | Core abstractions: Agent, ChatClient, Tools, Middleware, Session, Types |
| [`agent-framework-anthropic`](./crates/agent-framework-anthropic/) | Anthropic Claude provider (Messages API + SSE streaming) |
| [`agent-framework-openai`](./crates/agent-framework-openai/) | OpenAI provider (Chat Completions API + SSE streaming) |
| [`agent-framework`](./crates/agent-framework/) | Umbrella crate that re-exports everything with feature flags |

## Quick start

Add to your `Cargo.toml`:

```toml
[dependencies]
agent-framework = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

### Hello agent

```rust
use agent_framework::prelude::*;
use agent_framework::anthropic::{AnthropicChatClient, AnthropicConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AnthropicConfig::from_env()?;
    let agent = ChatClientAgent::builder()
        .client(AnthropicChatClient::new(config)?)
        .instructions("You are a helpful assistant.")
        .build()?;

    let mut session = AgentSession::new();
    let response = agent
        .run(vec![Message::user("Hello!")], &mut session, None)
        .await?;

    println!("{}", response.text);
    Ok(())
}
```

### Adding tools

```rust
use agent_framework::prelude::*;
use agent_framework_core::tools::{tool_fn, ToolDefinition};

let weather_tool = tool_fn(
    ToolDefinition::new(
        "get_weather",
        "Get current weather for a city",
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"]
        }),
    )?,
    |args| async move {
        let city = args["city"].as_str().unwrap_or("unknown");
        Ok(serde_json::json!({ "temperature": 72, "city": city, "condition": "sunny" }))
    },
);

let agent = ChatClientAgent::builder()
    .client(client)
    .instructions("Use the weather tool when asked about weather.")
    .tool(weather_tool)
    .build()?;
```

### Streaming

```rust
use tokio_stream::StreamExt;

let stream = agent.run_stream(
    vec![Message::user("Tell me a story")],
    &mut session,
    None,
)?;

// Print tokens as they arrive.
let mut stream = std::pin::pin!(stream);
while let Some(update) = stream.next().await {
    if let Some(text) = update?.text {
        print!("{text}");
    }
}
```

### Per-call option overrides

```rust
use agent_framework_core::types::{AgentRunOptions, ChatOptions, ToolChoice};

let run_opts = AgentRunOptions::new()
    .with_chat_options(ChatOptions {
        temperature: Some(0.0),
        tool_choice: Some(ToolChoice::Required),
        ..Default::default()
    })
    .with_additional_instructions("Respond in JSON only.");

let response = agent
    .run(vec![Message::user("What's 2+2?")], &mut session, Some(&run_opts))
    .await?;
```

### Middleware

```rust
use agent_framework_core::middleware::{
    AgentMiddleware, AgentMiddlewareNext, FunctionMiddleware, FunctionMiddlewareNext, MiddlewarePipeline,
};

struct LoggingAgentMiddleware;

#[async_trait::async_trait]
impl AgentMiddleware for LoggingAgentMiddleware {
    async fn on_run(
        &self,
        messages: Vec<Message>,
        session: &mut AgentSession,
        next: &dyn AgentMiddlewareNext,
    ) -> AgentResult<AgentResponse> {
        println!("Agent run starting with {} messages", messages.len());
        let result = next.run(messages, session).await;
        println!("Agent run complete");
        result
    }
}

let mut pipeline = MiddlewarePipeline::new();
pipeline.add_agent_middleware(LoggingAgentMiddleware);

let agent = ChatClientAgent::builder()
    .client(client)
    .middleware(pipeline)
    .build()?;
```

## Architecture

The Rust SDK mirrors the abstractions found in the Python and .NET SDKs:

```
                    +-------------------+
                    |   Agent trait      |  run(), run_stream()
                    +-------------------+
                            |
                   +--------+--------+
                   | ChatClientAgent  |  tool loop, middleware, session
                   +--------+--------+
                            |
                    +-------+-------+
                    | ChatClient    |  get_response(), get_response_stream()
                    +-------+-------+
                      /           \
          +-----------+    +----------+
          | Anthropic |    | OpenAI   |
          +-----------+    +----------+
```

- **`Agent` trait** — Core interface: `run` (non-streaming) and `run_stream` (streaming with tool loop).
- **`ChatClientAgent`** — Primary implementation backed by a `ChatClient`. Handles the model-call / tool-execute / model-continue loop, applies three-layer middleware, manages session history.
- **`ChatClient` trait** — Provider abstraction. Translate between framework types and provider APIs.
- **`FunctionTool` trait** — Tools the agent can invoke. Register via builder; the agent advertises them to the model and handles invocation.
- **Middleware** — Three layers applied in order:
  - *Agent middleware* wraps the entire `run` (logging, telemetry, guardrails).
  - *Chat-client middleware* wraps each model call via decorator pattern (request/response transforms).
  - *Function middleware* wraps each tool invocation (approval gates, auditing).
- **`AgentSession`** — Conversation state: history, arbitrary key-value state, optional `conversation_id` for server-managed threads.

## Samples

See [`samples/`](./samples/) for runnable examples organized by complexity:

| Sample | Description |
|--------|-------------|
| [01 - Hello Agent](./samples/01-get-started/hello_agent/) | Minimal single-turn conversation |
| [02 - Add Tools](./samples/02-add-tools/weather_agent/) | Register a function tool and let the agent use it |
| [03 - Multi-turn](./samples/03-multi-turn/conversation/) | Multi-turn conversation with session history |
| [04 - Streaming](./samples/04-streaming/stream_agent/) | Stream tokens to the terminal in real time |
| [05 - Middleware](./samples/05-middleware/logging/) | Add agent-level and function-level middleware |

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

## Security considerations

- **API keys**: Wrap in `SecretString` which redacts in `Debug`/`Display` and zeroizes on drop. Never log configs with `.api_key` exposed.
- **HTTP redirects**: Disabled via `Policy::none()`. The `x-api-key` header is *not* in reqwest's sensitive-header list and would leak on cross-host redirects if allowed.
- **Response size limits**: JSON bodies capped at 16 MiB; streaming state capped at 32 MiB.
- **Error scrubbing**: Provider error bodies are truncated and scrubbed for secret prefixes before embedding in `AgentError`.
- **Tool invocation**: The framework does *not* validate tool arguments against the declared JSON Schema before calling `FunctionTool::invoke`. Tools must validate their own inputs.
- **`InMemoryHistoryProvider`**: Development only. Capped at 1000 messages with FIFO eviction. Use a persistent `HistoryProvider` implementation for production.

## Requirements

- Rust 1.75+ (2021 edition)
- Tokio async runtime
