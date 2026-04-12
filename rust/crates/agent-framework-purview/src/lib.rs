// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework Purview
//!
//! Microsoft Purview governance integration for the Microsoft Agent Framework.
//!
//! Provides middleware that evaluates messages against Purview policy endpoints
//! before and after model calls, enabling content governance and compliance.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use agent_framework_core::client::ChatClient;
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::middleware::{AgentMiddleware, AgentMiddlewareNext, ChatClientMiddleware};
use agent_framework_core::secret::SecretString;
use agent_framework_core::session::AgentSession;
use agent_framework_core::streaming::ResponseStream;
use agent_framework_core::types::{AgentResponse, ChatOptions, ChatResponse, Message};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for connecting to a Microsoft Purview policy endpoint.
#[derive(Debug, Clone)]
pub struct PurviewConfig {
    /// The Purview policy endpoint URL.
    pub endpoint: String,
    /// Authentication token for the Purview endpoint.
    pub auth_token: SecretString,
}

impl PurviewConfig {
    /// Create a new configuration.
    pub fn new(endpoint: impl Into<String>, auth_token: impl Into<SecretString>) -> Self {
        Self {
            endpoint: endpoint.into(),
            auth_token: auth_token.into(),
        }
    }

    /// Create a configuration from environment variables.
    ///
    /// Reads `PURVIEW_ENDPOINT` and `PURVIEW_AUTH_TOKEN`.
    ///
    /// # Errors
    /// Returns an error if either variable is not set.
    pub fn from_env() -> AgentResult<Self> {
        let endpoint = std::env::var("PURVIEW_ENDPOINT").map_err(|_| {
            AgentError::InvalidRequest("PURVIEW_ENDPOINT environment variable not set".to_string())
        })?;
        let auth_token = std::env::var("PURVIEW_AUTH_TOKEN").map_err(|_| {
            AgentError::InvalidRequest("PURVIEW_AUTH_TOKEN environment variable not set".to_string())
        })?;
        Ok(Self {
            endpoint,
            auth_token: SecretString::new(auth_token),
        })
    }
}

// ---------------------------------------------------------------------------
// Policy request / response types
// ---------------------------------------------------------------------------

/// A policy evaluation request sent to the Purview endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRequest {
    /// The messages to evaluate.
    pub messages: Vec<Message>,
    /// The identifier of the agent making the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    /// The session identifier for tracking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// A policy evaluation response from the Purview endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyResponse {
    /// Whether the request is allowed by policy.
    pub allowed: bool,
    /// The reason for rejection, if not allowed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Modified messages if the policy rewrote content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modified_messages: Option<Vec<Message>>,
}

// ---------------------------------------------------------------------------
// Chat-client-level middleware
// ---------------------------------------------------------------------------

/// A [`ChatClientMiddleware`] that evaluates messages against a Purview
/// policy endpoint before each model call.
///
/// If the policy rejects the request, the middleware returns an error
/// instead of forwarding to the model. If the policy returns modified
/// messages, those are used in place of the originals.
pub struct PurviewMiddleware {
    config: PurviewConfig,
    http_client: reqwest::Client,
    /// Optional agent identifier sent with policy requests.
    pub agent_id: Option<String>,
    /// Optional session identifier sent with policy requests.
    pub session_id: Option<String>,
    /// Whether to also evaluate the model's response (post-evaluation).
    pub post_evaluate: bool,
}

impl PurviewMiddleware {
    /// Create a new Purview middleware with the given configuration.
    pub fn new(config: PurviewConfig) -> Self {
        Self {
            config,
            http_client: reqwest::Client::new(),
            agent_id: None,
            session_id: None,
            post_evaluate: false,
        }
    }

    /// Set the agent identifier for policy requests.
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Set the session identifier for policy requests.
    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Enable post-evaluation of model responses.
    pub fn with_post_evaluation(mut self, enabled: bool) -> Self {
        self.post_evaluate = enabled;
        self
    }

    /// Evaluate messages against the Purview policy endpoint.
    async fn evaluate_policy(&self, messages: &[Message]) -> AgentResult<PolicyResponse> {
        let request = PolicyRequest {
            messages: messages.to_vec(),
            agent_id: self.agent_id.clone(),
            session_id: self.session_id.clone(),
        };

        let url = format!("{}/evaluate", self.config.endpoint.trim_end_matches('/'));
        debug!(url = %url, "Evaluating policy with Purview");

        let response = self
            .http_client
            .post(&url)
            .bearer_auth(self.config.auth_token.expose())
            .json(&request)
            .send()
            .await
            .map_err(|e| AgentError::HttpError(format!("Purview request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(AgentError::HttpError(format!(
                "Purview endpoint returned {}: {}",
                status, body
            )));
        }

        response
            .json::<PolicyResponse>()
            .await
            .map_err(|e| AgentError::HttpError(format!("Failed to parse Purview response: {}", e)))
    }
}

#[async_trait]
impl ChatClientMiddleware for PurviewMiddleware {
    async fn on_get_response(
        &self,
        messages: &mut Vec<Message>,
        options: &mut ChatOptions,
        next: &dyn ChatClient,
    ) -> AgentResult<ChatResponse> {
        // Pre-evaluation: check messages against policy.
        let policy = self.evaluate_policy(messages).await?;

        if !policy.allowed {
            let reason = policy
                .reason
                .unwrap_or_else(|| "Request blocked by Purview policy".to_string());
            return Err(AgentError::InvalidRequest(reason));
        }

        // Apply modified messages if the policy rewrote them.
        if let Some(modified) = policy.modified_messages {
            *messages = modified;
        }

        // Proceed to the model.
        let response = next.get_response(messages, Some(options)).await?;

        // Post-evaluation: optionally check the model's response.
        if self.post_evaluate {
            let response_messages: Vec<Message> = response.messages.to_vec();

            let post_policy = self.evaluate_policy(&response_messages).await?;
            if !post_policy.allowed {
                let reason = post_policy
                    .reason
                    .unwrap_or_else(|| "Response blocked by Purview policy".to_string());
                warn!(reason = %reason, "Purview post-evaluation rejected model response");
                return Err(AgentError::InvalidResponse(reason));
            }
        }

        Ok(response)
    }

    async fn on_get_response_stream(
        &self,
        messages: &mut Vec<Message>,
        options: &mut ChatOptions,
        next: &dyn ChatClient,
    ) -> AgentResult<ResponseStream> {
        // Pre-evaluation: check messages against policy.
        let policy = self.evaluate_policy(messages).await?;

        if !policy.allowed {
            let reason = policy
                .reason
                .unwrap_or_else(|| "Request blocked by Purview policy".to_string());
            return Err(AgentError::InvalidRequest(reason));
        }

        // Apply modified messages if the policy rewrote them.
        if let Some(modified) = policy.modified_messages {
            *messages = modified;
        }

        // Proceed to the model. Post-evaluation is not applied for streaming
        // because the full response is not available until the stream completes.
        next.get_response_stream(messages, Some(options)).await
    }
}

// ---------------------------------------------------------------------------
// Agent-level middleware
// ---------------------------------------------------------------------------

/// An [`AgentMiddleware`] that applies Purview governance at the agent level.
///
/// Evaluates the input messages before the agent run and optionally evaluates
/// the agent's response after completion.
pub struct PurviewAgentMiddleware {
    config: PurviewConfig,
    http_client: reqwest::Client,
    /// Whether to also evaluate the agent's response (post-evaluation).
    pub post_evaluate: bool,
}

impl PurviewAgentMiddleware {
    /// Create a new Purview agent middleware with the given configuration.
    pub fn new(config: PurviewConfig) -> Self {
        Self {
            config,
            http_client: reqwest::Client::new(),
            post_evaluate: false,
        }
    }

    /// Enable post-evaluation of agent responses.
    pub fn with_post_evaluation(mut self, enabled: bool) -> Self {
        self.post_evaluate = enabled;
        self
    }

    /// Evaluate messages against the Purview policy endpoint.
    async fn evaluate_policy(
        &self,
        messages: &[Message],
        session: &AgentSession,
    ) -> AgentResult<PolicyResponse> {
        let request = PolicyRequest {
            messages: messages.to_vec(),
            agent_id: None,
            session_id: Some(session.session_id.clone()),
        };

        let url = format!("{}/evaluate", self.config.endpoint.trim_end_matches('/'));
        debug!(url = %url, "Evaluating agent-level policy with Purview");

        let response = self
            .http_client
            .post(&url)
            .bearer_auth(self.config.auth_token.expose())
            .json(&request)
            .send()
            .await
            .map_err(|e| AgentError::HttpError(format!("Purview request failed: {}", e)))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(AgentError::HttpError(format!(
                "Purview endpoint returned {}: {}",
                status, body
            )));
        }

        response
            .json::<PolicyResponse>()
            .await
            .map_err(|e| AgentError::HttpError(format!("Failed to parse Purview response: {}", e)))
    }
}

#[async_trait]
impl AgentMiddleware for PurviewAgentMiddleware {
    async fn on_run(
        &self,
        messages: Vec<Message>,
        session: &mut AgentSession,
        next: &dyn AgentMiddlewareNext,
    ) -> AgentResult<AgentResponse> {
        // Pre-evaluation: check input messages against policy.
        let policy = self.evaluate_policy(&messages, session).await?;

        if !policy.allowed {
            let reason = policy
                .reason
                .unwrap_or_else(|| "Request blocked by Purview policy".to_string());
            return Err(AgentError::InvalidRequest(reason));
        }

        // Apply modified messages if the policy rewrote them.
        let run_messages = policy.modified_messages.unwrap_or(messages);

        // Proceed to the next middleware or the agent.
        let response = next.run(run_messages, session).await?;

        // Post-evaluation: optionally check the agent's response.
        if self.post_evaluate {
            let post_policy = self.evaluate_policy(&response.messages, session).await?;
            if !post_policy.allowed {
                let reason = post_policy
                    .reason
                    .unwrap_or_else(|| "Response blocked by Purview policy".to_string());
                warn!(reason = %reason, "Purview post-evaluation rejected agent response");
                return Err(AgentError::InvalidResponse(reason));
            }
        }

        Ok(response)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_request_serialization() {
        let request = PolicyRequest {
            messages: vec![
                Message::user("Hello, can you help me?"),
                Message::assistant("Of course! How can I assist you?"),
            ],
            agent_id: Some("test-agent".to_string()),
            session_id: Some("session-123".to_string()),
        };

        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["agent_id"], "test-agent");
        assert_eq!(json["session_id"], "session-123");
        assert_eq!(json["messages"].as_array().unwrap().len(), 2);

        // Round-trip.
        let deserialized: PolicyRequest = serde_json::from_value(json).unwrap();
        assert_eq!(deserialized.agent_id.as_deref(), Some("test-agent"));
        assert_eq!(deserialized.messages.len(), 2);
    }

    #[test]
    fn policy_response_allowed() {
        let json = r#"{"allowed": true}"#;
        let response: PolicyResponse = serde_json::from_str(json).unwrap();
        assert!(response.allowed);
        assert!(response.reason.is_none());
        assert!(response.modified_messages.is_none());
    }

    #[test]
    fn policy_response_rejected() {
        let json = r#"{"allowed": false, "reason": "Content violates data protection policy"}"#;
        let response: PolicyResponse = serde_json::from_str(json).unwrap();
        assert!(!response.allowed);
        assert_eq!(
            response.reason.as_deref(),
            Some("Content violates data protection policy")
        );
    }

    #[test]
    fn policy_response_with_modified_messages() {
        let response = PolicyResponse {
            allowed: true,
            reason: None,
            modified_messages: Some(vec![Message::user("Sanitized message")]),
        };

        let json = serde_json::to_string(&response).unwrap();
        let deserialized: PolicyResponse = serde_json::from_str(&json).unwrap();
        assert!(deserialized.allowed);
        assert_eq!(deserialized.modified_messages.as_ref().unwrap().len(), 1);
        assert_eq!(
            deserialized.modified_messages.unwrap()[0].text(),
            "Sanitized message"
        );
    }

    #[test]
    fn policy_request_without_optional_fields() {
        let request = PolicyRequest {
            messages: vec![Message::user("test")],
            agent_id: None,
            session_id: None,
        };

        let json = serde_json::to_value(&request).unwrap();
        // Optional fields with skip_serializing_if should not be present.
        assert!(!json.as_object().unwrap().contains_key("agent_id"));
        assert!(!json.as_object().unwrap().contains_key("session_id"));
    }

    #[test]
    fn purview_config_new() {
        let config = PurviewConfig::new("https://purview.example.com", "token-123");
        assert_eq!(config.endpoint, "https://purview.example.com");
        assert_eq!(config.auth_token.expose(), "token-123");
    }
}
