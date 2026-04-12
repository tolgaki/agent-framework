// Copyright (c) Microsoft. All rights reserved.

//! Workflow state management.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// Shared workflow state that flows between nodes.
///
/// State is a thread-safe key-value store backed by `serde_json::Value`.
/// Nodes read inputs and write outputs through this shared state.
///
/// Mirrors .NET's workflow `State` and Python's `WorkflowContext`.
#[derive(Clone)]
pub struct WorkflowState {
    inner: Arc<RwLock<HashMap<String, serde_json::Value>>>,
}

impl WorkflowState {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create state from an initial set of values.
    pub fn from_map(map: HashMap<String, serde_json::Value>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(map)),
        }
    }

    /// Get a value by key.
    pub async fn get(&self, key: &str) -> Option<serde_json::Value> {
        self.inner.read().await.get(key).cloned()
    }

    /// Get a value and deserialize it into a typed struct.
    pub async fn get_as<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        let value = self.get(key).await?;
        serde_json::from_value(value).ok()
    }

    /// Set a value.
    pub async fn set(&self, key: impl Into<String>, value: serde_json::Value) {
        self.inner.write().await.insert(key.into(), value);
    }

    /// Remove a value.
    pub async fn remove(&self, key: &str) -> Option<serde_json::Value> {
        self.inner.write().await.remove(key)
    }

    /// Check if a key exists.
    pub async fn contains(&self, key: &str) -> bool {
        self.inner.read().await.contains_key(key)
    }

    /// Get all keys.
    pub async fn keys(&self) -> Vec<String> {
        self.inner.read().await.keys().cloned().collect()
    }

    /// Snapshot the entire state as a serializable map.
    pub async fn snapshot(&self) -> HashMap<String, serde_json::Value> {
        self.inner.read().await.clone()
    }

    /// Restore state from a snapshot.
    pub async fn restore(&self, snapshot: HashMap<String, serde_json::Value>) {
        *self.inner.write().await = snapshot;
    }
}

impl Default for WorkflowState {
    fn default() -> Self {
        Self::new()
    }
}

/// A serializable checkpoint of workflow execution state.
///
/// Enables suspend/resume of long-running workflows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowCheckpoint {
    /// The workflow run ID.
    pub run_id: String,
    /// State snapshot at checkpoint time.
    pub state: HashMap<String, serde_json::Value>,
    /// The set of completed node IDs.
    pub completed_nodes: Vec<String>,
    /// The next node(s) to execute.
    pub pending_nodes: Vec<String>,
}

/// Storage backend for workflow checkpoints.
#[async_trait::async_trait]
pub trait CheckpointStorage: Send + Sync {
    /// Save a checkpoint.
    async fn save(&self, checkpoint: &WorkflowCheckpoint) -> agent_framework_core::error::AgentResult<()>;
    /// Load the latest checkpoint for a run.
    async fn load(
        &self,
        run_id: &str,
    ) -> agent_framework_core::error::AgentResult<Option<WorkflowCheckpoint>>;
}

/// In-memory checkpoint storage for development and testing.
pub struct InMemoryCheckpointStorage {
    store: Arc<RwLock<HashMap<String, WorkflowCheckpoint>>>,
}

impl InMemoryCheckpointStorage {
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryCheckpointStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl CheckpointStorage for InMemoryCheckpointStorage {
    async fn save(&self, checkpoint: &WorkflowCheckpoint) -> agent_framework_core::error::AgentResult<()> {
        self.store
            .write()
            .await
            .insert(checkpoint.run_id.clone(), checkpoint.clone());
        Ok(())
    }

    async fn load(
        &self,
        run_id: &str,
    ) -> agent_framework_core::error::AgentResult<Option<WorkflowCheckpoint>> {
        Ok(self.store.read().await.get(run_id).cloned())
    }
}

/// A serializable file-based checkpoint storage.
pub struct FileCheckpointStorage {
    dir: std::path::PathBuf,
}

impl FileCheckpointStorage {
    pub fn new(dir: impl Into<std::path::PathBuf>) -> Self {
        Self { dir: dir.into() }
    }
}

#[async_trait::async_trait]
impl CheckpointStorage for FileCheckpointStorage {
    async fn save(&self, checkpoint: &WorkflowCheckpoint) -> agent_framework_core::error::AgentResult<()> {
        let path = self.dir.join(format!("{}.json", checkpoint.run_id));
        let json = serde_json::to_string_pretty(checkpoint)?;
        tokio::fs::create_dir_all(&self.dir)
            .await
            .map_err(|e| agent_framework_core::error::AgentError::Other(Box::new(e)))?;
        tokio::fs::write(path, json)
            .await
            .map_err(|e| agent_framework_core::error::AgentError::Other(Box::new(e)))?;
        Ok(())
    }

    async fn load(
        &self,
        run_id: &str,
    ) -> agent_framework_core::error::AgentResult<Option<WorkflowCheckpoint>> {
        let path = self.dir.join(format!("{run_id}.json"));
        match tokio::fs::read_to_string(&path).await {
            Ok(json) => {
                let cp = serde_json::from_str(&json)?;
                Ok(Some(cp))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(agent_framework_core::error::AgentError::Other(Box::new(e))),
        }
    }
}
