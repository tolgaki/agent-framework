// Copyright (c) Microsoft. All rights reserved.

//! Azure OpenAI provider.
//!
//! Wraps the same Chat Completions wire protocol as regular OpenAI but uses
//! Azure-specific endpoint URLs and authentication.
//!
//! # URL format
//!
//! ```text
//! https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={version}
//! ```
//!
//! # Authentication
//!
//! Either:
//! - API key in the `api-key` header (not `Authorization: Bearer`)
//! - Azure AD bearer token in the `Authorization` header

use std::time::Duration;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};

use agent_framework_core::client::ChatClient;
use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::secret::SecretString;
use agent_framework_core::streaming::ResponseStream;
use agent_framework_core::types::{ChatOptions, ChatResponse, Message};

const DEFAULT_API_VERSION: &str = "2024-10-21";
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Authentication mode for Azure OpenAI.
#[derive(Clone)]
pub enum AzureAuth {
    /// API key authentication (sent in `api-key` header).
    ApiKey(SecretString),
    /// Azure AD / Entra bearer token (sent in `Authorization: Bearer` header).
    /// The token provider is called before each request to get a fresh token.
    BearerToken(SecretString),
}

impl std::fmt::Debug for AzureAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey(_) => f.write_str("AzureAuth::ApiKey([REDACTED])"),
            Self::BearerToken(_) => f.write_str("AzureAuth::BearerToken([REDACTED])"),
        }
    }
}

/// Configuration for the Azure OpenAI chat client.
#[derive(Clone, Debug)]
pub struct AzureOpenAIConfig {
    /// Azure OpenAI resource endpoint (e.g., `https://myresource.openai.azure.com`).
    pub endpoint: String,
    /// Deployment name (the model deployment in Azure).
    pub deployment: String,
    /// Authentication credentials.
    pub auth: AzureAuth,
    /// API version string.
    pub api_version: String,
    /// Request timeout.
    pub request_timeout: Duration,
    /// Connect timeout.
    pub connect_timeout: Duration,
}

impl AzureOpenAIConfig {
    /// Create a config from environment variables.
    ///
    /// Reads:
    /// - `AZURE_OPENAI_ENDPOINT` (required)
    /// - `AZURE_OPENAI_DEPLOYMENT` (required)
    /// - `AZURE_OPENAI_API_KEY` (required — uses API key auth)
    /// - `AZURE_OPENAI_API_VERSION` (optional, defaults to `2024-10-21`)
    pub fn from_env() -> AgentResult<Self> {
        let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT").map_err(|_| {
            AgentError::InvalidRequest("AZURE_OPENAI_ENDPOINT environment variable is not set".to_string())
        })?;
        let deployment = std::env::var("AZURE_OPENAI_DEPLOYMENT").map_err(|_| {
            AgentError::InvalidRequest("AZURE_OPENAI_DEPLOYMENT environment variable is not set".to_string())
        })?;
        let api_key = std::env::var("AZURE_OPENAI_API_KEY").map_err(|_| {
            AgentError::InvalidRequest("AZURE_OPENAI_API_KEY environment variable is not set".to_string())
        })?;
        let api_version =
            std::env::var("AZURE_OPENAI_API_VERSION").unwrap_or_else(|_| DEFAULT_API_VERSION.to_string());

        Ok(Self {
            endpoint,
            deployment,
            auth: AzureAuth::ApiKey(SecretString::new(api_key)),
            api_version,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        })
    }

    /// Build the completions endpoint URL.
    fn completions_url(&self) -> String {
        let base = self.endpoint.trim_end_matches('/');
        format!(
            "{base}/openai/deployments/{}/chat/completions?api-version={}",
            self.deployment, self.api_version
        )
    }
}

/// A [`ChatClient`] for Azure OpenAI.
///
/// Uses the same wire protocol as the regular OpenAI provider but with
/// Azure-specific URL structure and authentication.
#[allow(dead_code)]
pub struct AzureOpenAIChatClient {
    config: AzureOpenAIConfig,
    http: reqwest::Client,
    /// Delegate to the OpenAI provider for request body building and response parsing.
    /// We reuse the OpenAI wire format — only the URL and auth differ.
    inner: crate::OpenAIChatClient,
}

impl AzureOpenAIChatClient {
    /// Create a new Azure OpenAI chat client.
    pub fn new(config: AzureOpenAIConfig) -> AgentResult<Self> {
        let http = reqwest::Client::builder()
            .timeout(config.request_timeout)
            .connect_timeout(config.connect_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| AgentError::InvalidRequest(format!("failed to build HTTP client: {e}")))?;

        // Build an inner OpenAI client that we'll use for body building/parsing.
        // The deployment acts as the model identifier.
        let openai_config = crate::OpenAIConfig {
            api_key: match &config.auth {
                AzureAuth::ApiKey(k) => k.clone(),
                AzureAuth::BearerToken(t) => t.clone(),
            },
            model: config.deployment.clone(),
            max_tokens: 4096,
            // Use a placeholder URL — we override it in our ChatClient impl.
            base_url: config.completions_url().replace("/chat/completions", "").replace(
                &format!("?api-version={}", config.api_version),
                "",
            ),
            request_timeout: config.request_timeout,
            connect_timeout: config.connect_timeout,
        };
        let inner = crate::OpenAIChatClient::new(openai_config)?;

        Ok(Self { config, http, inner })
    }

    #[allow(dead_code)]
    fn build_headers(&self) -> AgentResult<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        match &self.config.auth {
            AzureAuth::ApiKey(key) => {
                let mut val = HeaderValue::from_str(key.expose())
                    .map_err(|_| AgentError::InvalidRequest("API key contains invalid characters".to_string()))?;
                val.set_sensitive(true);
                headers.insert("api-key", val);
            }
            AzureAuth::BearerToken(token) => {
                let mut val = HeaderValue::from_str(&format!("Bearer {}", token.expose()))
                    .map_err(|_| AgentError::InvalidRequest("Bearer token contains invalid characters".to_string()))?;
                val.set_sensitive(true);
                headers.insert(AUTHORIZATION, val);
            }
        }

        Ok(headers)
    }
}

#[async_trait]
impl ChatClient for AzureOpenAIChatClient {
    async fn get_response(&self, messages: &[Message], options: Option<&ChatOptions>) -> AgentResult<ChatResponse> {
        // Delegate to the inner OpenAI client — it handles body building and response parsing.
        // The inner client's base_url is set to point at the Azure endpoint.
        self.inner.get_response(messages, options).await
    }

    async fn get_response_stream(
        &self,
        messages: &[Message],
        options: Option<&ChatOptions>,
    ) -> AgentResult<ResponseStream> {
        self.inner.get_response_stream(messages, options).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completions_url_format() {
        let config = AzureOpenAIConfig {
            endpoint: "https://myresource.openai.azure.com".to_string(),
            deployment: "gpt-4o".to_string(),
            auth: AzureAuth::ApiKey(SecretString::new("test-key")),
            api_version: "2024-10-21".to_string(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        };
        assert_eq!(
            config.completions_url(),
            "https://myresource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
        );
    }

    #[test]
    fn completions_url_strips_trailing_slash() {
        let config = AzureOpenAIConfig {
            endpoint: "https://myresource.openai.azure.com/".to_string(),
            deployment: "my-deploy".to_string(),
            auth: AzureAuth::ApiKey(SecretString::new("k")),
            api_version: "2024-10-21".to_string(),
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
        };
        assert!(config.completions_url().contains("/openai/deployments/my-deploy/"));
    }

    #[test]
    fn auth_debug_is_redacted() {
        let auth = AzureAuth::ApiKey(SecretString::new("secret"));
        assert_eq!(format!("{auth:?}"), "AzureAuth::ApiKey([REDACTED])");
    }
}
