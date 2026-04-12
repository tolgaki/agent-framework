// Copyright (c) Microsoft. All rights reserved.

//! Semantic convention constants for GenAI telemetry.
//!
//! Field names follow the [OpenTelemetry GenAI semantic conventions][otel-genai]
//! so that spans emitted by this crate are interoperable with standard OTel
//! collectors and dashboards.
//!
//! [otel-genai]: https://opentelemetry.io/docs/specs/semconv/gen-ai/

// ---------------------------------------------------------------------------
// GenAI system attributes
// ---------------------------------------------------------------------------

/// The GenAI system or provider (e.g. `"openai"`, `"anthropic"`).
pub const GEN_AI_SYSTEM: &str = "gen_ai.system";

/// The model name requested by the caller.
pub const GEN_AI_REQUEST_MODEL: &str = "gen_ai.request.model";

/// The model name reported in the response (may differ due to routing/aliasing).
pub const GEN_AI_RESPONSE_MODEL: &str = "gen_ai.response.model";

// ---------------------------------------------------------------------------
// Token usage attributes
// ---------------------------------------------------------------------------

/// Number of input (prompt) tokens consumed.
pub const GEN_AI_USAGE_INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";

/// Number of output (completion) tokens generated.
pub const GEN_AI_USAGE_OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";

// ---------------------------------------------------------------------------
// Operation attributes
// ---------------------------------------------------------------------------

/// The logical operation name (e.g. `"agent.run"`, `"chat.completion"`, `"tool.invoke"`).
pub const GEN_AI_OPERATION_NAME: &str = "gen_ai.operation.name";

/// The wall-clock duration of the operation in milliseconds.
pub const GEN_AI_OPERATION_DURATION: &str = "gen_ai.operation.duration_ms";

// ---------------------------------------------------------------------------
// Agent-specific attributes (framework extension)
// ---------------------------------------------------------------------------

/// The unique identifier of the agent instance.
pub const AGENT_ID: &str = "agent.id";

/// The human-readable name of the agent instance.
pub const AGENT_NAME: &str = "agent.name";
