// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework Orchestrations
//!
//! Multi-agent orchestration patterns. Mirrors .NET's workflow builders and
//! Python's `agent_framework_orchestrations` package.

mod concurrent;
mod group_chat;
mod handoff;
mod sequential;

pub use concurrent::*;
pub use group_chat::*;
pub use handoff::*;
pub use sequential::*;
