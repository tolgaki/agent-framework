// Copyright (c) Microsoft. All rights reserved.

//! Structured output helpers for typed agent responses.
//!
//! Mirrors .NET's `RunAsync<T>()` and Python's Pydantic structured output.
//! Wraps a normal agent run but sets `response_format` to JSON schema and
//! deserializes the model's text response into a concrete Rust type.

use serde::de::DeserializeOwned;

use crate::agent::Agent;
use crate::error::{AgentError, AgentResult};
use crate::session::AgentSession;
use crate::types::{AgentResponse, AgentRunOptions, ChatOptions, Message, ResponseFormat};

/// A typed agent response that wraps the raw `AgentResponse` with a
/// deserialized value of type `T`.
///
/// Corresponds to .NET's `AgentResponse<T>`.
#[derive(Debug, Clone)]
pub struct TypedAgentResponse<T> {
    /// The deserialized structured output.
    pub value: T,
    /// The underlying raw agent response (messages, usage, finish reason).
    pub raw: AgentResponse,
}

/// Run an agent and deserialize the response text into a typed struct.
///
/// This is the Rust equivalent of .NET's `agent.RunAsync<T>(...)`.
///
/// The function:
/// 1. Injects `response_format = JsonSchema` into the chat options based on
///    the provided JSON schema.
/// 2. Runs the agent normally.
/// 3. Deserializes the response text as JSON into `T`.
///
/// # Example
///
/// ```rust,no_run
/// use serde::Deserialize;
/// use agent_framework_core::structured_output::run_structured;
///
/// #[derive(Deserialize)]
/// struct Weather {
///     city: String,
///     temperature_f: f64,
///     condition: String,
/// }
///
/// # async fn example(agent: &dyn agent_framework_core::agent::Agent) {
/// let schema = serde_json::json!({
///     "type": "object",
///     "properties": {
///         "city": { "type": "string" },
///         "temperature_f": { "type": "number" },
///         "condition": { "type": "string" }
///     },
///     "required": ["city", "temperature_f", "condition"],
///     "additionalProperties": false
/// });
///
/// let mut session = agent_framework_core::session::AgentSession::new();
/// let typed = run_structured::<Weather>(
///     agent,
///     vec![agent_framework_core::types::Message::user("Weather in Seattle?")],
///     &mut session,
///     "Weather",
///     schema,
///     None,
/// ).await.unwrap();
///
/// println!("{} is {}°F", typed.value.city, typed.value.temperature_f);
/// # }
/// ```
pub async fn run_structured<T: DeserializeOwned>(
    agent: &dyn Agent,
    messages: Vec<Message>,
    session: &mut AgentSession,
    schema_name: &str,
    schema: serde_json::Value,
    options: Option<&AgentRunOptions>,
) -> AgentResult<TypedAgentResponse<T>> {
    // Build per-call options that force JSON schema response format.
    let json_format = ResponseFormat::JsonSchema {
        name: schema_name.to_string(),
        schema,
        strict: Some(true),
    };

    let run_options = match options {
        Some(opts) => {
            let mut chat = opts.chat_options.clone().unwrap_or_default();
            chat.response_format = Some(json_format);
            AgentRunOptions {
                chat_options: Some(chat),
                additional_instructions: opts.additional_instructions.clone(),
            }
        }
        None => AgentRunOptions {
            chat_options: Some(ChatOptions {
                response_format: Some(json_format),
                ..Default::default()
            }),
            additional_instructions: None,
        },
    };

    let response = agent.run(messages, session, Some(&run_options)).await?;

    let value: T = serde_json::from_str(&response.text).map_err(|e| {
        AgentError::InvalidResponse(format!(
            "failed to deserialize structured output as {}: {e}",
            std::any::type_name::<T>()
        ))
    })?;

    Ok(TypedAgentResponse { value, raw: response })
}
