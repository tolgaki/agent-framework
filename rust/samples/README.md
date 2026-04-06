# Agent Framework for Rust — Samples

Runnable examples organized by complexity, mirroring the Python and .NET SDK sample structure.

## Prerequisites

1. **Rust toolchain** — install via [rustup](https://rustup.rs/).
2. **API key** — set the relevant environment variable before running any sample:

| Provider | Environment variable | Example |
|----------|---------------------|---------|
| Anthropic | `ANTHROPIC_API_KEY` | `export ANTHROPIC_API_KEY="sk-ant-..."` |
| OpenAI | `OPENAI_API_KEY` | `export OPENAI_API_KEY="sk-..."` |

## Samples

### 01 — Get Started

| Sample | Command | Description |
|--------|---------|-------------|
| [Hello Agent](./01-get-started/hello_agent/) | `cargo run -p hello-agent` | Minimal single-turn agent conversation with Anthropic. |

### 02 — Add Tools

| Sample | Command | Description |
|--------|---------|-------------|
| [Weather Agent](./02-add-tools/weather_agent/) | `cargo run -p weather-agent` | Register a `get_weather` function tool; agent calls it automatically when asked about weather. |

### 03 — Multi-turn Conversations

| Sample | Command | Description |
|--------|---------|-------------|
| [Conversation](./03-multi-turn/conversation/) | `cargo run -p conversation` | Three-turn conversation where the agent remembers the user's name and preferences across turns via `AgentSession`. |

### 04 — Streaming

| Sample | Command | Description |
|--------|---------|-------------|
| [Stream Agent](./04-streaming/stream_agent/) | `cargo run -p stream-agent` | Stream tokens to the terminal in real time using `run_stream()`. |

### 05 — Middleware

| Sample | Command | Description |
|--------|---------|-------------|
| [Logging Middleware](./05-middleware/logging/) | `cargo run -p logging-middleware` | Add timing middleware at the agent level and function level to observe execution flow. |

## Running a sample

From the `rust/` directory:

```sh
export ANTHROPIC_API_KEY="sk-ant-..."
cargo run -p hello-agent
```

## Writing new samples

Follow the existing pattern:

1. Create a directory under the appropriate category (`NN-topic/sample_name/`).
2. Add `Cargo.toml` with `publish = false` and path dependencies to workspace crates.
3. Register the sample in the root `Cargo.toml` workspace members list.
4. Add a module-level doc comment at the top of `main.rs` explaining the sample and how to run it.
