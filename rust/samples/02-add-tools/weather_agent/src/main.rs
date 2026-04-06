// Copyright (c) Microsoft. All rights reserved.

//! # Weather Agent
//!
//! Demonstrates registering a function tool and letting the agent invoke it
//! automatically when the model decides it needs weather data.
//!
//! ## Prerequisites
//!
//! ```sh
//! export ANTHROPIC_API_KEY="sk-ant-..."
//! cargo run -p weather-agent
//! ```

use agent_framework_anthropic::{AnthropicChatClient, AnthropicConfig};
use agent_framework_core::agent::{Agent, ChatClientAgent};
use agent_framework_core::session::AgentSession;
use agent_framework_core::tools::{tool_fn, ToolDefinition};
use agent_framework_core::types::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = AnthropicConfig::from_env()?;

    // Define a tool that the model can call.
    let weather_tool = tool_fn(
        ToolDefinition::new(
            "get_weather",
            "Get the current weather for a city.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g. 'Seattle'"
                    }
                },
                "required": ["city"]
            }),
        ),
        |args| async move {
            let city = args["city"].as_str().unwrap_or("unknown");
            // In a real app this would call an API. Here we return mock data.
            Ok(serde_json::json!({
                "city": city,
                "temperature_f": 68,
                "condition": "partly cloudy"
            }))
        },
    );

    let agent = ChatClientAgent::builder()
        .client(AnthropicChatClient::new(config)?)
        .name("WeatherAgent")
        .instructions("You help users check the weather. Use the get_weather tool when asked.")
        .tool(weather_tool)
        .build()?;

    let mut session = AgentSession::new();
    let response = agent
        .run(
            vec![Message::user("What's the weather like in Seattle and Tokyo?")],
            &mut session,
            None,
        )
        .await?;

    println!("{}", response.text);
    Ok(())
}
