// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework
//!
//! The Microsoft Agent Framework for Rust.
//!
//! This is the umbrella crate that re-exports the core framework and all
//! enabled provider and extension crates.
//!
//! # Features
//!
//! **Providers** (default: anthropic + openai):
//! - `anthropic` — Anthropic Claude provider
//! - `openai` — OpenAI provider
//! - `azure-openai` — Azure OpenAI (enables `openai` + Azure auth)
//! - `ollama` — Ollama local LLM provider
//!
//! **Framework extensions:**
//! - `evaluation` — Agent evaluation framework
//! - `telemetry` — OpenTelemetry instrumentation
//! - `mcp` — Model Context Protocol support
//! - `workflows` — DAG workflow engine
//! - `orchestrations` — Multi-agent orchestration patterns
//! - `declarative` — JSON/YAML agent definitions
//!
//! **Persistence:**
//! - `cosmos` — Azure Cosmos DB persistence
//! - `redis` — Redis persistence
//!
//! **Hosting & integration:**
//! - `hosting` — HTTP server for agents (OpenAI-compatible API)
//! - `a2a` — Agent-to-Agent communication protocol
//! - `devui` — Development UI server
//! - `agui` — AG-UI (Agentic UI) protocol
//! - `purview` — Microsoft Purview governance
//!
//! - `all` — Enable everything

// Re-export core.
pub use agent_framework_core::*;

// --- Providers ---

/// Anthropic Claude provider.
#[cfg(feature = "anthropic")]
pub mod anthropic {
    pub use agent_framework_anthropic::*;
}

/// OpenAI provider.
#[cfg(feature = "openai")]
pub mod openai {
    pub use agent_framework_openai::*;
}

/// Ollama local LLM provider.
#[cfg(feature = "ollama")]
pub mod ollama {
    pub use agent_framework_ollama::*;
}

// --- Framework extensions ---

/// Agent evaluation framework.
#[cfg(feature = "evaluation")]
pub mod evaluation {
    pub use agent_framework_evaluation::*;
}

/// OpenTelemetry instrumentation.
#[cfg(feature = "telemetry")]
pub mod telemetry {
    pub use agent_framework_telemetry::*;
}

/// Model Context Protocol support.
#[cfg(feature = "mcp")]
pub mod mcp {
    pub use agent_framework_mcp::*;
}

/// DAG workflow engine.
#[cfg(feature = "workflows")]
pub mod workflows {
    pub use agent_framework_workflows::*;
}

/// Multi-agent orchestration patterns.
#[cfg(feature = "orchestrations")]
pub mod orchestrations {
    pub use agent_framework_orchestrations::*;
}

/// Declarative agent definitions.
#[cfg(feature = "declarative")]
pub mod declarative {
    pub use agent_framework_declarative::*;
}

// --- Persistence ---

/// Azure Cosmos DB persistence.
#[cfg(feature = "cosmos")]
pub mod cosmos {
    pub use agent_framework_cosmos::*;
}

/// Redis persistence.
#[cfg(feature = "redis")]
pub mod redis {
    pub use agent_framework_redis::*;
}

// --- Hosting & integration ---

/// HTTP server hosting.
#[cfg(feature = "hosting")]
pub mod hosting {
    pub use agent_framework_hosting::*;
}

/// Agent-to-Agent communication.
#[cfg(feature = "a2a")]
pub mod a2a {
    pub use agent_framework_a2a::*;
}

/// Development UI server.
#[cfg(feature = "devui")]
pub mod devui {
    pub use agent_framework_devui::*;
}

/// AG-UI protocol support.
#[cfg(feature = "agui")]
pub mod agui {
    pub use agent_framework_agui::*;
}

/// Microsoft Purview governance.
#[cfg(feature = "purview")]
pub mod purview {
    pub use agent_framework_purview::*;
}

/// Azure AI Foundry agent provider.
#[cfg(feature = "foundry")]
pub mod foundry {
    pub use agent_framework_foundry::*;
}

/// Azure AI Persistent Agents.
#[cfg(feature = "azure-persistent")]
pub mod azure_persistent {
    pub use agent_framework_azure_persistent::*;
}

/// Microsoft Copilot Studio integration.
#[cfg(feature = "copilot-studio")]
pub mod copilot_studio {
    pub use agent_framework_copilot_studio::*;
}

/// GitHub Copilot agent provider.
#[cfg(feature = "github-copilot")]
pub mod github_copilot {
    pub use agent_framework_github_copilot::*;
}

/// Mem0 long-term memory provider.
#[cfg(feature = "mem0")]
pub mod mem0 {
    pub use agent_framework_mem0::*;
}

/// Durable task execution.
#[cfg(feature = "durabletask")]
pub mod durabletask {
    pub use agent_framework_durabletask::*;
}

/// Azure Functions hosting.
#[cfg(feature = "hosting-azfunc")]
pub mod hosting_azfunc {
    pub use agent_framework_hosting_azfunc::*;
}

/// Declarative YAML/JSON workflow definitions.
#[cfg(feature = "workflows-declarative")]
pub mod workflows_declarative {
    pub use agent_framework_workflows_declarative::*;
}

/// Common imports for quick starts.
pub mod prelude {
    pub use agent_framework_core::agent::{Agent, ChatClientAgent};
    pub use agent_framework_core::client::ChatClient;
    pub use agent_framework_core::error::{AgentError, AgentResult};
    pub use agent_framework_core::secret::SecretString;
    pub use agent_framework_core::session::AgentSession;
    pub use agent_framework_core::tools::{FunctionTool, ToolDefinition};
    pub use agent_framework_core::types::{
        AgentResponse, ChatOptions, Content, Message, ResponseFormat, Role, Usage,
    };
}
