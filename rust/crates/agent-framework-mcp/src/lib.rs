// Copyright (c) Microsoft. All rights reserved.

//! # Agent Framework MCP
//!
//! Model Context Protocol (MCP) support for the Microsoft Agent Framework.
//!
//! This crate allows agents to discover and invoke tools exposed by MCP-compliant
//! servers. It mirrors the Python SDK's `_mcp.py` module, providing:
//!
//! - [`McpServer`] — connects to an MCP server, discovers tools, and invokes them.
//! - [`StdioTransport`] — communicates with a subprocess over stdin/stdout JSON-RPC.
//! - [`McpTool`] — wraps a single MCP tool as a [`FunctionTool`].
//! - [`McpStdioTool`] — convenience constructor that spawns a process and returns
//!   all discovered tools.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use agent_framework_mcp::McpStdioTool;
//!
//! # async fn example() -> agent_framework_core::AgentResult<()> {
//! let tools = McpStdioTool::connect("npx", &["-y", "@anthropic/mcp-server-example"]).await?;
//! // `tools` is a Vec<Box<dyn FunctionTool>> ready to pass to an agent.
//! # Ok(())
//! # }
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::Mutex;
use tracing::{debug, warn};

use agent_framework_core::error::{AgentError, AgentResult};
use agent_framework_core::tools::{FunctionTool, ToolDefinition};

// ---------------------------------------------------------------------------
// JSON-RPC types (internal wire format)
// ---------------------------------------------------------------------------

/// A JSON-RPC 2.0 request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// JSON-RPC version, always "2.0".
    pub jsonrpc: String,
    /// Request identifier.
    pub id: u64,
    /// The method name to invoke.
    pub method: String,
    /// Parameters for the method.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    /// Create a new JSON-RPC 2.0 request.
    pub fn new(id: u64, method: impl Into<String>, params: Option<Value>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            method: method.into(),
            params,
        }
    }
}

/// A JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// JSON-RPC version, always "2.0".
    pub jsonrpc: String,
    /// The request identifier this response corresponds to.
    pub id: u64,
    /// The result on success.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// The error on failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// A JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// A numeric error code.
    pub code: i64,
    /// A short human-readable description.
    pub message: String,
}

// ---------------------------------------------------------------------------
// McpTransport trait
// ---------------------------------------------------------------------------

/// Transport abstraction for sending JSON-RPC messages to an MCP server.
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Send a JSON-RPC request and wait for the corresponding response.
    async fn send(&self, request: JsonRpcRequest) -> AgentResult<JsonRpcResponse>;

    /// Gracefully close the transport.
    async fn close(&self) -> AgentResult<()>;
}

// ---------------------------------------------------------------------------
// StdioTransport
// ---------------------------------------------------------------------------

/// Internal mutable state for [`StdioTransport`].
struct StdioState {
    stdin: ChildStdin,
    stdout_reader: BufReader<ChildStdout>,
    child: Child,
}

/// A transport that communicates with a subprocess via stdin/stdout.
///
/// Each JSON-RPC message is a single line of JSON terminated by `\n`.
pub struct StdioTransport {
    state: Mutex<Option<StdioState>>,
    next_id: AtomicU64,
}

impl StdioTransport {
    /// Spawn a new subprocess and return a transport connected to it.
    ///
    /// # Errors
    /// Returns an error if the subprocess cannot be spawned.
    pub fn new(command: &str, args: &[&str]) -> AgentResult<Self> {
        debug!(command, ?args, "spawning MCP stdio subprocess");

        let mut child = tokio::process::Command::new(command)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| AgentError::HttpError(format!("failed to spawn MCP process: {e}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| AgentError::HttpError("failed to open stdin for MCP process".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AgentError::HttpError("failed to open stdout for MCP process".into()))?;

        Ok(Self {
            state: Mutex::new(Some(StdioState {
                stdin,
                stdout_reader: BufReader::new(stdout),
                child,
            })),
            next_id: AtomicU64::new(1),
        })
    }

    /// Allocate the next request ID.
    #[allow(dead_code)]
    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn send(&self, request: JsonRpcRequest) -> AgentResult<JsonRpcResponse> {
        let mut guard = self.state.lock().await;
        let state = guard
            .as_mut()
            .ok_or_else(|| AgentError::InvalidRequest("MCP transport is closed".into()))?;

        // Serialize request as a single JSON line.
        let mut line = serde_json::to_string(&request)?;
        line.push('\n');

        debug!(id = request.id, method = %request.method, "sending JSON-RPC request");

        state
            .stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|e| AgentError::HttpError(format!("failed to write to MCP stdin: {e}")))?;
        state
            .stdin
            .flush()
            .await
            .map_err(|e| AgentError::HttpError(format!("failed to flush MCP stdin: {e}")))?;

        // Read lines until we get a JSON-RPC response matching our request ID.
        // MCP servers may emit notifications (no `id`) which we skip.
        loop {
            let mut response_line = String::new();
            let bytes_read = state
                .stdout_reader
                .read_line(&mut response_line)
                .await
                .map_err(|e| AgentError::HttpError(format!("failed to read from MCP stdout: {e}")))?;

            if bytes_read == 0 {
                return Err(AgentError::HttpError(
                    "MCP process closed stdout unexpectedly".into(),
                ));
            }

            let trimmed = response_line.trim();
            if trimmed.is_empty() {
                continue;
            }

            // Try to parse as a JSON-RPC response.
            match serde_json::from_str::<JsonRpcResponse>(trimmed) {
                Ok(resp) if resp.id == request.id => {
                    debug!(id = resp.id, "received JSON-RPC response");
                    return Ok(resp);
                }
                Ok(resp) => {
                    // Response for a different ID — this shouldn't happen in a
                    // simple sequential protocol, but log and skip.
                    warn!(
                        expected_id = request.id,
                        actual_id = resp.id,
                        "received JSON-RPC response with unexpected id, skipping"
                    );
                }
                Err(_) => {
                    // Could be a JSON-RPC notification (no `id` field) or server log output.
                    debug!(line = trimmed, "skipping non-response line from MCP server");
                }
            }
        }
    }

    async fn close(&self) -> AgentResult<()> {
        let mut guard = self.state.lock().await;
        if let Some(mut state) = guard.take() {
            debug!("closing MCP stdio transport");
            // Drop stdin to signal EOF to the child process.
            drop(state.stdin);
            // Wait briefly for the child to exit, then kill if needed.
            match tokio::time::timeout(std::time::Duration::from_secs(5), state.child.wait()).await {
                Ok(Ok(status)) => {
                    debug!(?status, "MCP process exited");
                }
                Ok(Err(e)) => {
                    warn!("error waiting for MCP process: {e}");
                }
                Err(_) => {
                    warn!("MCP process did not exit within timeout, killing");
                    let _ = state.child.kill().await;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// McpToolInfo
// ---------------------------------------------------------------------------

/// Metadata about a tool discovered from an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolInfo {
    /// The tool's name as reported by the MCP server.
    pub name: String,
    /// A human-readable description of the tool.
    #[serde(default)]
    pub description: String,
    /// The JSON Schema describing the tool's input parameters.
    #[serde(default = "default_schema", rename = "inputSchema")]
    pub input_schema: Value,
}

fn default_schema() -> Value {
    serde_json::json!({
        "type": "object",
        "properties": {},
    })
}

// ---------------------------------------------------------------------------
// McpServer
// ---------------------------------------------------------------------------

/// An MCP server connection that can discover and invoke tools.
///
/// Wraps a [`McpTransport`] and provides high-level operations for the MCP
/// protocol (initialize, list tools, call tool).
pub struct McpServer {
    transport: Arc<Mutex<Box<dyn McpTransport>>>,
    next_id: AtomicU64,
}

impl McpServer {
    /// Create a new MCP server wrapper around a transport.
    pub fn new(transport: Box<dyn McpTransport>) -> Self {
        Self {
            transport: Arc::new(Mutex::new(transport)),
            next_id: AtomicU64::new(1),
        }
    }

    /// Allocate the next request ID.
    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Send a JSON-RPC request through the transport and return the result.
    async fn request(&self, method: &str, params: Option<Value>) -> AgentResult<Value> {
        let id = self.next_id();
        let request = JsonRpcRequest::new(id, method, params);

        let transport = self.transport.lock().await;
        let response = transport.send(request).await?;

        if let Some(err) = response.error {
            return Err(AgentError::HttpError(format!(
                "MCP JSON-RPC error {}: {}",
                err.code, err.message
            )));
        }

        Ok(response.result.unwrap_or(Value::Null))
    }

    /// Send the `initialize` request to the MCP server.
    ///
    /// This must be called before using any other methods. It negotiates
    /// the protocol version and capabilities.
    pub async fn initialize(&self) -> AgentResult<()> {
        let params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "agent-framework-rust",
                "version": env!("CARGO_PKG_VERSION"),
            }
        });

        let result = self.request("initialize", Some(params)).await?;
        debug!(?result, "MCP server initialized");

        // Send the `notifications/initialized` notification. The MCP spec says
        // this is a notification (no response expected), but our transport is
        // request-response. Some servers reply, some don't. We send it and
        // tolerate failures — the next real request will skip stale lines.
        let id = self.next_id();
        let notification = JsonRpcRequest::new(
            id,
            "notifications/initialized",
            Some(serde_json::json!({})),
        );
        let transport = self.transport.lock().await;
        let _ = transport.send(notification).await;

        Ok(())
    }

    /// List all tools available on the MCP server.
    pub async fn list_tools(&self) -> AgentResult<Vec<McpToolInfo>> {
        let result = self.request("tools/list", Some(serde_json::json!({}))).await?;

        // The result should have a "tools" array.
        let tools_value = result
            .get("tools")
            .cloned()
            .unwrap_or_else(|| Value::Array(vec![]));

        let tools: Vec<McpToolInfo> = serde_json::from_value(tools_value).map_err(|e| {
            AgentError::InvalidResponse(format!("failed to parse MCP tools/list response: {e}"))
        })?;

        debug!(count = tools.len(), "discovered MCP tools");
        Ok(tools)
    }

    /// Call a tool on the MCP server.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> AgentResult<Value> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments,
        });

        let result = self.request("tools/call", Some(params)).await?;
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// McpTool — implements FunctionTool
// ---------------------------------------------------------------------------

/// A single MCP tool exposed as a [`FunctionTool`].
///
/// Created by [`McpStdioTool`] or manually from an [`McpServer`] and
/// [`McpToolInfo`].
pub struct McpTool {
    server: Arc<McpServer>,
    definition: ToolDefinition,
    remote_name: String,
}

impl McpTool {
    /// Create a new `McpTool` wrapping a specific tool on the given server.
    ///
    /// # Errors
    /// Returns an error if the tool name fails validation.
    pub fn new(server: Arc<McpServer>, info: &McpToolInfo) -> AgentResult<Self> {
        // Normalize the remote name for use as the tool definition name.
        let normalized = normalize_mcp_name(&info.name);
        let definition = ToolDefinition::new(
            &normalized,
            &info.description,
            info.input_schema.clone(),
        )?;

        Ok(Self {
            server,
            definition,
            remote_name: info.name.clone(),
        })
    }
}

#[async_trait]
impl FunctionTool for McpTool {
    fn definition(&self) -> &ToolDefinition {
        &self.definition
    }

    async fn invoke(&self, args: Value) -> AgentResult<Value> {
        self.server.call_tool(&self.remote_name, args).await
    }
}

/// Normalize an MCP tool name to the allowed identifier pattern `[a-zA-Z0-9_-]`.
///
/// Characters outside the allowed set are replaced with `-`. This mirrors the
/// Python SDK's `_normalize_mcp_name`.
fn normalize_mcp_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// McpStdioTool — convenience constructor
// ---------------------------------------------------------------------------

/// Convenience constructor that spawns an MCP subprocess, initializes the
/// connection, discovers tools, and returns them as [`FunctionTool`] trait objects.
///
/// This mirrors the Python SDK's `MCPStdioTool`.
///
/// # Example
///
/// ```rust,no_run
/// use agent_framework_mcp::McpStdioTool;
///
/// # async fn example() -> agent_framework_core::AgentResult<()> {
/// let tools = McpStdioTool::connect("npx", &["-y", "@modelcontextprotocol/server-filesystem"]).await?;
/// println!("Discovered {} tools", tools.len());
/// # Ok(())
/// # }
/// ```
pub struct McpStdioTool;

impl McpStdioTool {
    /// Spawn an MCP subprocess, initialize it, and return all discovered tools.
    ///
    /// # Errors
    /// Returns an error if the subprocess cannot be spawned, initialization
    /// fails, or tool discovery fails.
    pub async fn connect(
        command: &str,
        args: &[&str],
    ) -> AgentResult<Vec<Box<dyn FunctionTool>>> {
        let transport = StdioTransport::new(command, args)?;
        let server = Arc::new(McpServer::new(Box::new(transport)));

        server.initialize().await?;
        let tool_infos = server.list_tools().await?;

        let mut tools: Vec<Box<dyn FunctionTool>> = Vec::with_capacity(tool_infos.len());
        for info in &tool_infos {
            match McpTool::new(Arc::clone(&server), info) {
                Ok(tool) => tools.push(Box::new(tool)),
                Err(e) => {
                    warn!(tool_name = %info.name, error = %e, "skipping MCP tool with invalid name");
                }
            }
        }

        debug!(count = tools.len(), "MCP stdio tools ready");
        Ok(tools)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- JSON-RPC serialization tests --

    #[test]
    fn json_rpc_request_serialization() {
        let req = JsonRpcRequest::new(1, "initialize", Some(serde_json::json!({"foo": "bar"})));
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("\"method\":\"initialize\""));
        assert!(json.contains("\"params\""));

        // Round-trip.
        let parsed: JsonRpcRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, 1);
        assert_eq!(parsed.method, "initialize");
        assert_eq!(parsed.jsonrpc, "2.0");
    }

    #[test]
    fn json_rpc_request_no_params() {
        let req = JsonRpcRequest::new(42, "ping", None);
        let json = serde_json::to_string(&req).unwrap();
        // `params` should be omitted when None.
        assert!(!json.contains("\"params\""));

        let parsed: JsonRpcRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, 42);
        assert!(parsed.params.is_none());
    }

    #[test]
    fn json_rpc_response_success() {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, 1);
        assert!(resp.result.is_some());
        assert!(resp.error.is_none());
    }

    #[test]
    fn json_rpc_response_error() {
        let json = r#"{"jsonrpc":"2.0","id":2,"error":{"code":-32601,"message":"Method not found"}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.id, 2);
        assert!(resp.result.is_none());
        let err = resp.error.unwrap();
        assert_eq!(err.code, -32601);
        assert_eq!(err.message, "Method not found");
    }

    #[test]
    fn json_rpc_error_serialization() {
        let err = JsonRpcError {
            code: -32600,
            message: "Invalid Request".into(),
        };
        let json = serde_json::to_string(&err).unwrap();
        let parsed: JsonRpcError = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.code, -32600);
        assert_eq!(parsed.message, "Invalid Request");
    }

    // -- McpToolInfo tests --

    #[test]
    fn mcp_tool_info_deserialization() {
        let json = r#"{
            "name": "read_file",
            "description": "Read a file from disk",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }
        }"#;
        let info: McpToolInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.name, "read_file");
        assert_eq!(info.description, "Read a file from disk");
        assert!(info.input_schema.get("properties").is_some());
    }

    #[test]
    fn mcp_tool_info_missing_optional_fields() {
        let json = r#"{"name": "ping"}"#;
        let info: McpToolInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.name, "ping");
        assert_eq!(info.description, "");
        assert_eq!(info.input_schema, default_schema());
    }

    // -- Name normalization tests --

    #[test]
    fn normalize_mcp_name_basic() {
        assert_eq!(normalize_mcp_name("read_file"), "read_file");
        assert_eq!(normalize_mcp_name("my-tool"), "my-tool");
        assert_eq!(normalize_mcp_name("tool.with.dots"), "tool-with-dots");
        assert_eq!(normalize_mcp_name("hello world!"), "hello-world-");
        assert_eq!(normalize_mcp_name("CamelCase123"), "CamelCase123");
    }

    // -- Fake / Mock transports for testing --

    /// A fake transport that always returns an error. Used for synchronous
    /// construction tests.
    struct FakeTransport;

    #[async_trait]
    impl McpTransport for FakeTransport {
        async fn send(&self, _request: JsonRpcRequest) -> AgentResult<JsonRpcResponse> {
            Err(AgentError::HttpError("fake transport".into()))
        }

        async fn close(&self) -> AgentResult<()> {
            Ok(())
        }
    }

    /// A mock transport that returns canned responses in FIFO order.
    struct MockTransport {
        responses: Mutex<Vec<JsonRpcResponse>>,
    }

    impl MockTransport {
        fn new(responses: Vec<JsonRpcResponse>) -> Self {
            Self {
                responses: Mutex::new(responses),
            }
        }
    }

    #[async_trait]
    impl McpTransport for MockTransport {
        async fn send(&self, request: JsonRpcRequest) -> AgentResult<JsonRpcResponse> {
            let mut responses = self.responses.lock().await;
            if responses.is_empty() {
                return Err(AgentError::HttpError("no more mock responses".into()));
            }
            let mut resp = responses.remove(0);
            // Match the response id to the request id.
            resp.id = request.id;
            Ok(resp)
        }

        async fn close(&self) -> AgentResult<()> {
            Ok(())
        }
    }

    fn ok_response(result: Value) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: 0, // overridden by MockTransport
            result: Some(result),
            error: None,
        }
    }

    fn err_response(code: i64, message: &str) -> JsonRpcResponse {
        JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: 0,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.to_string(),
            }),
        }
    }

    // -- McpTool construction tests --

    #[test]
    fn mcp_tool_definition_from_info() {
        let server = Arc::new(McpServer::new(Box::new(FakeTransport)));
        let info = McpToolInfo {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        };
        let tool = McpTool::new(server, &info).unwrap();
        let def = tool.definition();
        assert_eq!(def.name, "test_tool");
        assert_eq!(def.description, "A test tool");
    }

    // -- McpServer async tests --

    #[tokio::test]
    async fn mcp_server_request_error_handling() {
        let server = McpServer::new(Box::new(FakeTransport));
        let result = server.initialize().await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("fake transport"));
    }

    #[tokio::test]
    async fn mcp_server_initialize_and_list_tools() {
        let transport = MockTransport::new(vec![
            // Response to "initialize"
            ok_response(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "serverInfo": {"name": "test-server", "version": "1.0"}
            })),
            // Response to "notifications/initialized"
            ok_response(serde_json::json!({})),
            // Response to "tools/list"
            ok_response(serde_json::json!({
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"}
                            }
                        }
                    },
                    {
                        "name": "write_file",
                        "description": "Write a file",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"}
                            }
                        }
                    }
                ]
            })),
        ]);

        let server = McpServer::new(Box::new(transport));
        server.initialize().await.unwrap();

        let tools = server.list_tools().await.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "read_file");
        assert_eq!(tools[1].name, "write_file");
    }

    #[tokio::test]
    async fn mcp_server_call_tool() {
        let transport = MockTransport::new(vec![ok_response(serde_json::json!({
            "content": [{"type": "text", "text": "file contents here"}]
        }))]);

        let server = McpServer::new(Box::new(transport));
        let result = server
            .call_tool("read_file", serde_json::json!({"path": "/tmp/test.txt"}))
            .await
            .unwrap();

        assert!(result.get("content").is_some());
    }

    #[tokio::test]
    async fn mcp_server_json_rpc_error() {
        let transport = MockTransport::new(vec![err_response(-32601, "Method not found")]);

        let server = McpServer::new(Box::new(transport));
        let result = server.request("nonexistent", None).await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("-32601"));
        assert!(err_msg.contains("Method not found"));
    }

    #[tokio::test]
    async fn mcp_tool_invoke() {
        let transport = MockTransport::new(vec![ok_response(
            serde_json::json!({"result": "success"}),
        )]);

        let server = Arc::new(McpServer::new(Box::new(transport)));
        let info = McpToolInfo {
            name: "test_tool".to_string(),
            description: "A test".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        };

        let tool = McpTool::new(server, &info).unwrap();
        let result = tool.invoke(serde_json::json!({})).await.unwrap();
        assert_eq!(result, serde_json::json!({"result": "success"}));
    }
}
