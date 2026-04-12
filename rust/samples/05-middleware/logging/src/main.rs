// Copyright (c) Microsoft. All rights reserved.

//! # Middleware Example
//!
//! Demonstrates all three middleware layers:
//!
//! 1. **Agent middleware** — wraps the entire `run()`, logging entry/exit.
//! 2. **Function middleware** — wraps each tool invocation, timing execution.
//! 3. **Chat-client middleware** — wraps each model call (not shown here but
//!    applied the same way via `MiddlewarePipeline::add_chat_client_middleware`).
//!
//! ## Prerequisites
//!
//! ```sh
//! export ANTHROPIC_API_KEY="sk-ant-..."
//! cargo run -p logging-middleware
//! ```

use std::time::Instant;

use async_trait::async_trait;

use agent_framework_anthropic::{AnthropicChatClient, AnthropicConfig};
use agent_framework_core::agent::{Agent, ChatClientAgent};
use agent_framework_core::error::AgentResult;
use agent_framework_core::middleware::{
    AgentMiddleware, AgentMiddlewareNext, FunctionMiddleware, FunctionMiddlewareNext, MiddlewarePipeline,
};
use agent_framework_core::session::AgentSession;
use agent_framework_core::tools::{tool_fn, FunctionTool, ToolDefinition};
use agent_framework_core::types::{AgentResponse, Message};

// --- Agent-level middleware ---

struct TimingAgentMiddleware;

#[async_trait]
impl AgentMiddleware for TimingAgentMiddleware {
    async fn on_run(
        &self,
        messages: Vec<Message>,
        session: &mut AgentSession,
        next: &dyn AgentMiddlewareNext,
    ) -> AgentResult<AgentResponse> {
        let start = Instant::now();
        println!("[agent] run starting ({} input messages)", messages.len());
        let result = next.run(messages, session).await;
        println!("[agent] run finished in {:?}", start.elapsed());
        result
    }
}

// --- Function-level middleware ---

struct TimingFunctionMiddleware;

#[async_trait]
impl FunctionMiddleware for TimingFunctionMiddleware {
    async fn on_invoke(
        &self,
        tool: &dyn FunctionTool,
        args: serde_json::Value,
        next: &dyn FunctionMiddlewareNext,
    ) -> AgentResult<serde_json::Value> {
        let name = tool.definition().name.clone();
        let start = Instant::now();
        println!("[tool] invoking '{name}'");
        let result = next.invoke(args).await;
        println!("[tool] '{name}' finished in {:?}", start.elapsed());
        result
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AnthropicConfig::from_env()?;

    let calculator = tool_fn(
        ToolDefinition::new(
            "calculate",
            "Evaluate a math expression",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "expression": { "type": "string", "description": "e.g. '2 + 3 * 4'" }
                },
                "required": ["expression"]
            }),
        )?,
        |args| async move {
            let expr = args["expression"].as_str().unwrap_or("0");
            // Toy evaluation — in a real app, use a proper math parser.
            Ok(serde_json::json!({ "expression": expr, "result": 42 }))
        },
    );

    let mut pipeline = MiddlewarePipeline::new();
    pipeline.add_agent_middleware(TimingAgentMiddleware);
    pipeline.add_function_middleware(TimingFunctionMiddleware);

    let agent = ChatClientAgent::builder()
        .client(AnthropicChatClient::new(config)?)
        .name("MiddlewareAgent")
        .instructions("Use the calculate tool for any math questions.")
        .tool(calculator)
        .middleware(pipeline)
        .build()?;

    let mut session = AgentSession::new();
    let response = agent
        .run(vec![Message::user("What is 17 * 24 + 3?")], &mut session, None)
        .await?;

    println!("\nAssistant: {}", response.text);
    Ok(())
}
