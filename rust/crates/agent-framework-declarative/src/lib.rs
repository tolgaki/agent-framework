// Copyright (c) Microsoft. All rights reserved.

//! # Declarative Agents
//!
//! Define agents via JSON configuration files instead of code.
//! Mirrors .NET's `PromptAgentFactory` and Python's `AgentFactory`.
//!
//! # Example configuration
//!
//! ```json
//! {
//!   "name": "WeatherAgent",
//!   "instructions": "You help users check the weather.",
//!   "model": "claude-sonnet-4-20250514",
//!   "provider": "anthropic",
//!   "tools": ["get_weather"],
//!   "max_tool_rounds": 5
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use agent_framework_core::agent::ChatClientAgent;
use agent_framework_core::client::ChatClient;
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::tools::FunctionTool;
use agent_framework_core::types::ChatOptions;

/// A declarative agent definition loaded from JSON/YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDefinition {
    /// Agent name.
    pub name: String,

    /// System instructions.
    #[serde(default)]
    pub instructions: Option<String>,

    /// Model identifier.
    #[serde(default)]
    pub model: Option<String>,

    /// Provider name (e.g., "anthropic", "openai", "ollama").
    #[serde(default)]
    pub provider: Option<String>,

    /// Tool names to attach (resolved by the factory's tool registry).
    #[serde(default)]
    pub tools: Vec<String>,

    /// Maximum tool-calling rounds.
    #[serde(default)]
    pub max_tool_rounds: Option<usize>,

    /// Description.
    #[serde(default)]
    pub description: Option<String>,

    /// Extra provider-specific options.
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl AgentDefinition {
    /// Load a definition from a JSON string.
    pub fn from_json(json: &str) -> AgentResult<Self> {
        serde_json::from_str(json).map_err(|e| {
            AgentError::InvalidRequest(format!("failed to parse agent definition: {e}"))
        })
    }

    /// Load a definition from a JSON file.
    pub fn from_file(path: &std::path::Path) -> AgentResult<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            AgentError::InvalidRequest(format!("failed to read {}: {e}", path.display()))
        })?;
        Self::from_json(&content)
    }
}

/// A factory function that creates a [`ChatClient`] from a provider name and
/// optional model override.
pub type ProviderFactory =
    Box<dyn Fn(Option<&str>) -> AgentResult<Box<dyn ChatClient>> + Send + Sync>;

/// Factory for building agents from declarative definitions.
///
/// Register providers and tools, then call [`build`] with a definition to
/// create a configured agent.
///
/// Mirrors .NET's `PromptAgentFactory` and Python's `AgentFactory`.
pub struct AgentFactory {
    providers: HashMap<String, ProviderFactory>,
    tools: HashMap<String, Box<dyn Fn() -> Box<dyn FunctionTool> + Send + Sync>>,
}

impl AgentFactory {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            tools: HashMap::new(),
        }
    }

    /// Register a provider factory.
    pub fn register_provider(
        mut self,
        name: impl Into<String>,
        factory: impl Fn(Option<&str>) -> AgentResult<Box<dyn ChatClient>> + Send + Sync + 'static,
    ) -> Self {
        self.providers.insert(name.into(), Box::new(factory));
        self
    }

    /// Register a tool factory.
    pub fn register_tool(
        mut self,
        name: impl Into<String>,
        factory: impl Fn() -> Box<dyn FunctionTool> + Send + Sync + 'static,
    ) -> Self {
        self.tools.insert(name.into(), Box::new(factory));
        self
    }

    /// Build an agent from a definition.
    pub fn build(&self, definition: &AgentDefinition) -> AgentResult<ChatClientAgent> {
        let provider_name = definition
            .provider
            .as_deref()
            .unwrap_or("anthropic");

        let provider_factory = self.providers.get(provider_name).ok_or_else(|| {
            AgentError::InvalidRequest(format!("unknown provider: {provider_name}"))
        })?;

        let client = provider_factory(definition.model.as_deref())?;

        let mut builder = ChatClientAgent::builder()
            .client_boxed(client)
            .name(&definition.name);

        if let Some(desc) = &definition.description {
            builder = builder.description(desc);
        }

        if let Some(instructions) = &definition.instructions {
            builder = builder.instructions(instructions);
        }

        if let Some(max) = definition.max_tool_rounds {
            builder = builder.max_tool_rounds(max);
        }

        if let Some(model) = &definition.model {
            builder = builder.options(ChatOptions {
                model: Some(model.clone()),
                ..Default::default()
            });
        }

        let mut tools_vec: Vec<Box<dyn FunctionTool>> = Vec::new();
        for tool_name in &definition.tools {
            let tool_factory = self.tools.get(tool_name).ok_or_else(|| {
                AgentError::InvalidRequest(format!("unknown tool: {tool_name}"))
            })?;
            tools_vec.push(tool_factory());
        }
        if !tools_vec.is_empty() {
            builder = builder.tools(tools_vec);
        }

        builder.build()
    }

    /// Build an agent from a JSON string.
    pub fn build_from_json(&self, json: &str) -> AgentResult<ChatClientAgent> {
        let definition = AgentDefinition::from_json(json)?;
        self.build(&definition)
    }

    /// Build an agent from a JSON file.
    pub fn build_from_file(&self, path: &std::path::Path) -> AgentResult<ChatClientAgent> {
        let definition = AgentDefinition::from_file(path)?;
        self.build(&definition)
    }
}

impl Default for AgentFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_definition_from_json() {
        let json = r#"{
            "name": "TestAgent",
            "instructions": "Be helpful",
            "provider": "openai",
            "model": "gpt-4o",
            "tools": ["search", "calculator"],
            "max_tool_rounds": 5
        }"#;

        let def = AgentDefinition::from_json(json).unwrap();
        assert_eq!(def.name, "TestAgent");
        assert_eq!(def.provider.as_deref(), Some("openai"));
        assert_eq!(def.tools.len(), 2);
        assert_eq!(def.max_tool_rounds, Some(5));
    }

    #[test]
    fn minimal_definition() {
        let json = r#"{"name": "Minimal"}"#;
        let def = AgentDefinition::from_json(json).unwrap();
        assert_eq!(def.name, "Minimal");
        assert!(def.instructions.is_none());
        assert!(def.tools.is_empty());
    }
}
