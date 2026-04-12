// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework OpenAI
//!
//! OpenAI provider for the Microsoft Agent Framework.
//!
//! This crate provides [`OpenAIChatClient`], a [`ChatClient`](agent_framework_core::ChatClient)
//! implementation that communicates with the OpenAI Chat Completions API.

mod chat_client;

pub use chat_client::{OpenAIChatClient, OpenAIConfig};

/// Azure OpenAI provider (feature-gated).
#[cfg(feature = "azure")]
pub mod azure;
#[cfg(feature = "azure")]
pub use azure::{AzureAuth, AzureOpenAIChatClient, AzureOpenAIConfig};
