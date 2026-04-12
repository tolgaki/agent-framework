// Copyright (c) Microsoft. All rights reserved.

use async_trait::async_trait;
use tokio_stream::StreamExt;
use tracing::{debug, instrument};

use crate::client::ChatClient;
use crate::context::ContextProvider;
use crate::error::{AgentError, AgentResult};
use crate::http_limits::DEFAULT_MAX_STREAM_STATE_BYTES;
use crate::middleware::{
    AgentChain, AgentMiddleware, AgentMiddlewareNext, ChatClientDecorator, FunctionChain, FunctionMiddleware,
    FunctionMiddlewareNext, MiddlewarePipeline,
};
use crate::session::AgentSession;
use crate::streaming::AgentResponseStream;
use crate::tools::FunctionTool;
use crate::types::{
    AgentResponse, AgentResponseUpdate, AgentRunOptions, ChatOptions, Content, FinishReason, Message, Role, Usage,
};

/// The core agent trait.
///
/// An agent can participate in conversations, invoke tools, and manage sessions.
/// This corresponds to Python's `BaseAgent` and .NET's `AIAgent`.
#[async_trait]
pub trait Agent: Send + Sync {
    /// A unique identifier for this agent instance.
    fn id(&self) -> &str;

    /// An optional human-readable name.
    fn name(&self) -> Option<&str>;

    /// An optional description of the agent's purpose.
    fn description(&self) -> Option<&str>;

    /// Run the agent with the given input messages.
    ///
    /// `options` can override `ChatOptions` fields and add instructions for
    /// this call only; pass `None` to use the agent's configured defaults.
    async fn run(
        &self,
        messages: Vec<Message>,
        session: &mut AgentSession,
        options: Option<&AgentRunOptions>,
    ) -> AgentResult<AgentResponse>;

    /// Run the agent and return a stream of incremental updates.
    ///
    /// The returned stream borrows from `self` and `session` and must be
    /// fully consumed (or dropped) before either can be used again.
    ///
    /// **Note:** Agent-level middleware is NOT applied during streaming because
    /// [`AgentMiddleware::on_run`] returns an `AgentResponse` (not a stream).
    /// Chat-client middleware and function middleware are still applied.
    /// Use [`run`](Self::run) if you need agent-level middleware guarantees.
    fn run_stream<'a>(
        &'a self,
        messages: Vec<Message>,
        session: &'a mut AgentSession,
        options: Option<&'a AgentRunOptions>,
    ) -> AgentResult<AgentResponseStream<'a>>;
}

/// The primary agent implementation backed by a [`ChatClient`].
///
/// This is the workhorse agent that most users will interact with. It wraps
/// a chat client (LLM provider), manages tool execution loops, and applies
/// middleware. Corresponds to Python's `Agent` class and .NET's `ChatClientAgent`.
///
/// # Example
///
/// ```rust,no_run
/// use agent_framework_core::agent::{Agent, ChatClientAgent};
/// use agent_framework_core::types::Message;
/// use agent_framework_core::session::AgentSession;
///
/// # async fn example(client: Box<dyn agent_framework_core::client::ChatClient>) {
/// let agent = ChatClientAgent::builder()
///     .client_boxed(client)
///     .instructions("You are a helpful assistant.")
///     .build()
///     .unwrap();
///
/// let mut session = AgentSession::new();
/// let response = agent.run(vec![Message::user("Hello!")], &mut session, None).await.unwrap();
/// println!("{}", response.text);
/// # }
/// ```
pub struct ChatClientAgent {
    id: String,
    name: Option<String>,
    description: Option<String>,
    instructions: Option<String>,
    client: Box<dyn ChatClient>,
    tools: Vec<Box<dyn FunctionTool>>,
    context_providers: Vec<Box<dyn ContextProvider>>,
    options: ChatOptions,
    max_tool_rounds: usize,
    agent_middleware: Vec<Box<dyn AgentMiddleware>>,
    function_middleware: Vec<Box<dyn FunctionMiddleware>>,
}

impl ChatClientAgent {
    /// Create a builder for constructing a `ChatClientAgent`.
    pub fn builder() -> ChatClientAgentBuilder {
        ChatClientAgentBuilder::default()
    }

    /// Find a tool by name.
    fn find_tool(&self, name: &str) -> Option<&dyn FunctionTool> {
        self.tools
            .iter()
            .find(|t| t.definition().name == name)
            .map(|t| t.as_ref())
    }

    /// Build the full message list including system instructions and context,
    /// and collect any tools injected by context providers.
    async fn build_messages_and_context(
        &self,
        input_messages: &[Message],
        session: &AgentSession,
        additional_instructions: Option<&str>,
    ) -> AgentResult<(Vec<Box<dyn FunctionTool>>, Vec<Message>)> {
        let mut messages = Vec::new();
        let mut context_tools: Vec<Box<dyn FunctionTool>> = Vec::new();

        // Add system instructions.
        if let Some(instructions) = &self.instructions {
            messages.push(Message::system(instructions.clone()));
        }

        // Per-call additional instructions (from AgentRunOptions) stack on top.
        if let Some(extra) = additional_instructions {
            if !extra.is_empty() {
                messages.push(Message::system(extra.to_string()));
            }
        }

        // Add context from providers.
        for provider in &self.context_providers {
            let context = provider.provide_context(session).await?;
            if let Some(instructions) = context.instructions {
                messages.push(Message::system(instructions));
            }
            messages.extend(context.messages);
            context_tools.extend(context.tools);
        }

        // Add conversation history.
        let history = session.get_history().await?;
        messages.extend(history);

        // Add input messages.
        messages.extend(input_messages.iter().cloned());

        Ok((context_tools, messages))
    }

    /// Build the effective ChatOptions for one run by layering per-call
    /// overrides onto the agent's default options, then attaching tool
    /// definitions (agent-level + context-provided), deduplicating by name.
    fn effective_options(
        &self,
        per_call: Option<&AgentRunOptions>,
        context_tools: &[Box<dyn FunctionTool>],
    ) -> ChatOptions {
        let mut options = match per_call.and_then(|o| o.chat_options.as_ref()) {
            Some(overrides) => self.options.merge(overrides),
            None => self.options.clone(),
        };

        // Collect all tool definitions: agent tools + context tools.
        let mut seen = std::collections::HashSet::new();
        // Keep any already in options (from per-call overrides).
        for t in &options.tools {
            seen.insert(t.name.clone());
        }
        for t in &self.tools {
            let def = t.definition();
            if seen.insert(def.name.clone()) {
                options.tools.push(def.clone());
            }
        }
        for t in context_tools {
            let def = t.definition();
            if seen.insert(def.name.clone()) {
                options.tools.push(def.clone());
            }
        }
        options
    }

    /// Invoke a tool by name through the function middleware chain.
    /// Checks both agent-owned tools and dynamically injected context tools.
    async fn invoke_tool(
        &self,
        name: &str,
        args: serde_json::Value,
        context_tools: &[Box<dyn FunctionTool>],
    ) -> String {
        let tool = self.find_tool(name).or_else(|| {
            context_tools
                .iter()
                .find(|t| t.definition().name == name)
                .map(|t| t.as_ref())
        });
        match tool {
            Some(tool) => {
                let chain = FunctionChain {
                    remaining: &self.function_middleware,
                    tool,
                };
                match chain.invoke(args).await {
                    Ok(value) => match serde_json::to_string(&value) {
                        Ok(s) => s,
                        Err(e) => format!("Error: serialization failed: {e}"),
                    },
                    Err(e) => format!("Error: {e}"),
                }
            }
            None => format!("Error: unknown tool '{name}'"),
        }
    }

    /// Execute the tool-call loop: call model, handle tool calls, repeat.
    #[instrument(skip(self, session, run_options), fields(agent_id = %self.id))]
    async fn execute_run(
        &self,
        input_messages: Vec<Message>,
        session: &mut AgentSession,
        run_options: Option<&AgentRunOptions>,
    ) -> AgentResult<AgentResponse> {
        let additional = run_options.and_then(|o| o.additional_instructions.as_deref());
        let (context_tools, mut all_messages) =
            self.build_messages_and_context(&input_messages, session, additional).await?;
        let mut accumulated_messages = Vec::new();
        let options = self.effective_options(run_options, &context_tools);
        let mut total_usage = Usage::default();

        for round in 0..self.max_tool_rounds {
            debug!(round, "Invoking chat client");

            let response = self.client.get_response(&all_messages, Some(&options)).await?;

            // Accumulate usage across all model calls.
            if let Some(u) = &response.usage {
                total_usage.input_tokens += u.input_tokens;
                total_usage.output_tokens += u.output_tokens;
            }

            // Collect assistant messages.
            accumulated_messages.extend(response.messages.clone());

            // Check if any messages contain tool calls.
            let tool_calls: Vec<_> = response
                .messages
                .iter()
                .flat_map(|m| m.tool_calls())
                .map(|(id, name, args)| (id.to_string(), name.to_string(), args.clone()))
                .collect();

            if tool_calls.is_empty() || response.finish_reason != Some(FinishReason::ToolUse) {
                // No tool calls or model is done — save to history and return.
                let mut history_messages = input_messages;
                history_messages.extend(accumulated_messages.clone());
                session.save_history(&history_messages).await?;

                let mut agent_response = AgentResponse::from_chat_response(response, accumulated_messages);
                agent_response.usage = Some(total_usage);
                return Ok(agent_response);
            }

            // Add assistant messages (with tool calls) to the conversation.
            all_messages.extend(response.messages.clone());

            // Execute tool calls (each wrapped in function middleware).
            for (id, name, args) in tool_calls {
                debug!(tool = %name, "Invoking tool");
                let result = self.invoke_tool(&name, args, &context_tools).await;

                let tool_msg = Message {
                    role: Role::Tool,
                    content: vec![Content::tool_result(&id, result)],
                    name: Some(name),
                    metadata: Default::default(),
                };
                all_messages.push(tool_msg.clone());
                accumulated_messages.push(tool_msg);
            }

            debug!(round, "Tool round complete, continuing");
        }

        Err(AgentError::InvalidResponse(format!(
            "Exceeded maximum tool rounds ({})",
            self.max_tool_rounds
        )))
    }
}

/// Terminal node for the agent middleware chain that calls the real inner run.
struct ExecuteRunTerminal<'a> {
    agent: &'a ChatClientAgent,
    options: Option<&'a AgentRunOptions>,
}

#[async_trait]
impl<'a> AgentMiddlewareNext for ExecuteRunTerminal<'a> {
    async fn run(&self, messages: Vec<Message>, session: &mut AgentSession) -> AgentResult<AgentResponse> {
        self.agent.execute_run(messages, session, self.options).await
    }
}

#[async_trait]
impl Agent for ChatClientAgent {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    async fn run(
        &self,
        messages: Vec<Message>,
        session: &mut AgentSession,
        options: Option<&AgentRunOptions>,
    ) -> AgentResult<AgentResponse> {
        let terminal = ExecuteRunTerminal { agent: self, options };
        let chain = AgentChain {
            remaining: &self.agent_middleware,
            terminal: &terminal,
        };
        chain.run(messages, session).await
    }

    fn run_stream<'a>(
        &'a self,
        messages: Vec<Message>,
        session: &'a mut AgentSession,
        options: Option<&'a AgentRunOptions>,
    ) -> AgentResult<AgentResponseStream<'a>> {
        self.execute_run_stream(messages, session, options)
    }
}

// Streaming implementation lives in the Agent impl via a helper on ChatClientAgent.
// It cannot participate in agent middleware today because AgentMiddleware::on_run
// returns an AgentResponse (not a stream). Chat-client middleware still applies
// (via the decorator chain on the wrapped client), and function middleware still
// applies around tool invocations during the streaming loop.
impl ChatClientAgent {
    /// Run the agent and stream incremental updates.
    ///
    /// Drives the full tool-calling loop while streaming text deltas through
    /// to the caller. Tool-call argument fragments are accumulated silently
    /// and only invoked after the model signals `finish_reason = tool_use`.
    fn execute_run_stream<'a>(
        &'a self,
        input_messages: Vec<Message>,
        session: &'a mut AgentSession,
        run_options: Option<&'a AgentRunOptions>,
    ) -> AgentResult<AgentResponseStream<'a>> {
        let stream = async_stream::try_stream! {
            let additional = run_options.and_then(|o| o.additional_instructions.as_deref());
            let (context_tools, mut all_messages) =
                self.build_messages_and_context(&input_messages, session, additional).await?;
            let mut accumulated_messages: Vec<Message> = Vec::new();
            let options = self.effective_options(run_options, &context_tools);

            for round in 0..self.max_tool_rounds {
                debug!(round, "Opening chat stream");

                let mut provider_stream = self.client.get_response_stream(&all_messages, Some(&options)).await?;

                let mut accumulated_text = String::new();
                let mut tool_accumulators: Vec<StreamingToolCall> = Vec::new();
                let mut final_finish_reason: Option<FinishReason> = None;

                while let Some(update) = provider_stream.next().await {
                    let update = update?;

                    if let Some(text) = &update.text {
                        accumulated_text.push_str(text);
                    }
                    if let Some(tc) = &update.tool_call {
                        push_tool_call_delta(&mut tool_accumulators, tc);
                    }
                    if let Some(reason) = update.finish_reason {
                        final_finish_reason = Some(reason);
                    }

                    let state_bytes = accumulated_text.len()
                        + tool_accumulators.iter().map(|t| t.id.len() + t.name.len() + t.arguments.len()).sum::<usize>();
                    if state_bytes > DEFAULT_MAX_STREAM_STATE_BYTES {
                        Err::<(), _>(AgentError::InvalidResponse(format!(
                            "streaming state exceeded {} bytes",
                            DEFAULT_MAX_STREAM_STATE_BYTES
                        )))?;
                    }

                    yield AgentResponseUpdate {
                        text: update.text.clone(),
                        inner: update,
                    };
                }

                let assistant_msg = build_assistant_message(&accumulated_text, &tool_accumulators);
                accumulated_messages.push(assistant_msg.clone());
                all_messages.push(assistant_msg);

                let has_tool_calls = !tool_accumulators.is_empty();
                if !has_tool_calls || final_finish_reason != Some(FinishReason::ToolUse) {
                    let mut history_messages = input_messages.clone();
                    history_messages.extend(accumulated_messages.clone());
                    session.save_history(&history_messages).await?;
                    return;
                }

                // Execute tool calls through the function middleware chain.
                for tc in tool_accumulators {
                    let result = match serde_json::from_str::<serde_json::Value>(&tc.arguments) {
                        Err(e) => format!("Error: invalid JSON arguments: {e}"),
                        Ok(args) => self.invoke_tool(&tc.name, args, &context_tools).await,
                    };

                    let tool_msg = Message {
                        role: Role::Tool,
                        content: vec![Content::tool_result(&tc.id, result)],
                        name: Some(tc.name.clone()),
                        metadata: Default::default(),
                    };
                    all_messages.push(tool_msg.clone());
                    accumulated_messages.push(tool_msg);
                }

                debug!(round, "Tool round complete, continuing stream");
            }

            Err::<(), _>(AgentError::InvalidResponse(format!(
                "Exceeded maximum tool rounds ({})",
                self.max_tool_rounds
            )))?;
        };

        Ok(AgentResponseStream::new(stream))
    }
}

/// Accumulator for a single in-flight tool call during streaming.
struct StreamingToolCall {
    id: String,
    name: String,
    arguments: String,
}

fn push_tool_call_delta(acc: &mut Vec<StreamingToolCall>, delta: &crate::types::ToolCallUpdate) {
    // Match by id when available.
    if !delta.id.is_empty() {
        if let Some(existing) = acc.iter_mut().find(|t| t.id == delta.id) {
            if existing.name.is_empty() && !delta.name.is_empty() {
                existing.name = delta.name.clone();
            }
            existing.arguments.push_str(&delta.arguments_delta);
            return;
        }
    } else if let Some(existing) = acc.last_mut() {
        // Positional fallback: when id is empty (e.g., continuation chunks
        // from some providers), append to the most recently added accumulator.
        if existing.name.is_empty() && !delta.name.is_empty() {
            existing.name = delta.name.clone();
        }
        existing.arguments.push_str(&delta.arguments_delta);
        return;
    }
    // No match — start a new accumulator.
    acc.push(StreamingToolCall {
        id: delta.id.clone(),
        name: delta.name.clone(),
        arguments: delta.arguments_delta.clone(),
    });
}

fn build_assistant_message(text: &str, tools: &[StreamingToolCall]) -> Message {
    let mut content: Vec<Content> = Vec::new();
    if !text.is_empty() {
        content.push(Content::text(text));
    }
    for tc in tools {
        // If the streamed JSON is malformed, preserve the raw fragment as a
        // JSON string so the conversation record isn't silently falsified
        // with Null. The tool invocation path surfaces its own error.
        let args: serde_json::Value =
            serde_json::from_str(&tc.arguments).unwrap_or_else(|_| serde_json::Value::String(tc.arguments.clone()));
        content.push(Content::tool_call(&tc.id, &tc.name, args));
    }
    Message {
        role: Role::Assistant,
        content,
        name: None,
        metadata: Default::default(),
    }
}

/// Builder for [`ChatClientAgent`].
#[derive(Default)]
pub struct ChatClientAgentBuilder {
    name: Option<String>,
    description: Option<String>,
    instructions: Option<String>,
    client: Option<Box<dyn ChatClient>>,
    tools: Vec<Box<dyn FunctionTool>>,
    context_providers: Vec<Box<dyn ContextProvider>>,
    options: ChatOptions,
    max_tool_rounds: Option<usize>,
    middleware: Option<MiddlewarePipeline>,
}

impl ChatClientAgentBuilder {
    /// Set the chat client (LLM provider). Required.
    pub fn client(mut self, client: impl ChatClient + 'static) -> Self {
        self.client = Some(Box::new(client));
        self
    }

    /// Set the chat client from a boxed trait object.
    pub fn client_boxed(mut self, client: Box<dyn ChatClient>) -> Self {
        self.client = Some(client);
        self
    }

    /// Set the agent's human-readable name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the agent's description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the system instructions for the agent.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Add a tool to the agent.
    pub fn tool(mut self, tool: impl FunctionTool + 'static) -> Self {
        self.tools.push(Box::new(tool));
        self
    }

    /// Add multiple tools to the agent.
    pub fn tools(mut self, tools: Vec<Box<dyn FunctionTool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Add a context provider.
    pub fn context_provider(mut self, provider: impl ContextProvider + 'static) -> Self {
        self.context_providers.push(Box::new(provider));
        self
    }

    /// Set default chat options.
    pub fn options(mut self, options: ChatOptions) -> Self {
        self.options = options;
        self
    }

    /// Set the maximum number of tool-call rounds before giving up.
    pub fn max_tool_rounds(mut self, max: usize) -> Self {
        self.max_tool_rounds = Some(max);
        self
    }

    /// Set the middleware pipeline.
    pub fn middleware(mut self, pipeline: MiddlewarePipeline) -> Self {
        self.middleware = Some(pipeline);
        self
    }

    /// Build the [`ChatClientAgent`].
    ///
    /// # Errors
    /// Returns an error if no client was provided.
    pub fn build(self) -> AgentResult<ChatClientAgent> {
        let client = self
            .client
            .ok_or_else(|| AgentError::InvalidRequest("A ChatClient is required".to_string()))?;

        let pipeline = self.middleware.unwrap_or_default();

        // Wrap the client with chat-client middleware decorators. The first
        // middleware added becomes the outermost layer (called first), matching
        // AgentMiddleware and FunctionMiddleware ordering.
        let mut wrapped: Box<dyn ChatClient> = client;
        for mw in pipeline.chat_client_middleware.into_iter().rev() {
            wrapped = Box::new(ChatClientDecorator::new(mw, wrapped));
        }

        Ok(ChatClientAgent {
            id: uuid::Uuid::new_v4().to_string(),
            name: self.name,
            description: self.description,
            instructions: self.instructions,
            client: wrapped,
            tools: self.tools,
            context_providers: self.context_providers,
            options: self.options,
            max_tool_rounds: self.max_tool_rounds.unwrap_or(10),
            agent_middleware: pipeline.agent_middleware,
            function_middleware: pipeline.function_middleware,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatResponse;

    /// A mock chat client for testing.
    struct MockChatClient {
        response: ChatResponse,
    }

    #[async_trait]
    impl ChatClient for MockChatClient {
        async fn get_response(
            &self,
            _messages: &[Message],
            _options: Option<&ChatOptions>,
        ) -> AgentResult<ChatResponse> {
            Ok(self.response.clone())
        }

        async fn get_response_stream(
            &self,
            _messages: &[Message],
            _options: Option<&ChatOptions>,
        ) -> AgentResult<crate::streaming::ResponseStream> {
            Err(AgentError::InvalidRequest("Not implemented".to_string()))
        }
    }

    #[tokio::test]
    async fn test_basic_agent_run() {
        let mock_response = ChatResponse {
            messages: vec![Message::assistant("Hello! How can I help you?")],
            response_id: Some("test-123".to_string()),
            finish_reason: Some(FinishReason::Stop),
            usage: None,
        };

        let agent = ChatClientAgent::builder()
            .client(MockChatClient {
                response: mock_response,
            })
            .name("TestAgent")
            .instructions("You are a test assistant.")
            .build()
            .unwrap();

        assert_eq!(agent.name(), Some("TestAgent"));

        let mut session = AgentSession::new();
        let response = agent.run(vec![Message::user("Hi")], &mut session, None).await.unwrap();

        assert_eq!(response.text, "Hello! How can I help you?");
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn test_builder_requires_client() {
        let result = ChatClientAgent::builder().name("NoClient").build();
        assert!(result.is_err());
    }

    // ---------- Middleware tests ----------

    use crate::middleware::{ChatClientMiddleware, MiddlewarePipeline};
    use crate::tools::{tool_fn, ToolDefinition};
    use std::sync::Arc;
    use std::sync::Mutex;

    /// Mock client that returns different responses based on call count, to
    /// exercise the agent's tool-calling loop.
    struct ScriptedClient {
        responses: Arc<Mutex<Vec<ChatResponse>>>,
    }

    #[async_trait]
    impl ChatClient for ScriptedClient {
        async fn get_response(
            &self,
            _messages: &[Message],
            _options: Option<&ChatOptions>,
        ) -> AgentResult<ChatResponse> {
            let mut r = self.responses.lock().unwrap();
            if r.is_empty() {
                return Err(AgentError::InvalidResponse("no scripted response".into()));
            }
            Ok(r.remove(0))
        }
        async fn get_response_stream(
            &self,
            _: &[Message],
            _: Option<&ChatOptions>,
        ) -> AgentResult<crate::streaming::ResponseStream> {
            Err(AgentError::Unimplemented("stream".into()))
        }
    }

    /// Function middleware that records each invocation's args.
    struct RecordingFunctionMiddleware {
        label: &'static str,
        log: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl FunctionMiddleware for RecordingFunctionMiddleware {
        async fn on_invoke(
            &self,
            _tool: &dyn FunctionTool,
            args: serde_json::Value,
            next: &dyn FunctionMiddlewareNext,
        ) -> AgentResult<serde_json::Value> {
            self.log.lock().unwrap().push(format!("{}:before", self.label));
            let result = next.invoke(args).await;
            self.log.lock().unwrap().push(format!("{}:after", self.label));
            result
        }
    }

    #[tokio::test]
    async fn function_middleware_wraps_tool_invocation_in_order() {
        let log = Arc::new(Mutex::new(Vec::<String>::new()));

        // Scripted: first response requests a tool call; second returns final text.
        let responses = vec![
            ChatResponse {
                messages: vec![Message {
                    role: Role::Assistant,
                    content: vec![Content::tool_call("call_1", "echo", serde_json::json!({"msg": "hi"}))],
                    name: None,
                    metadata: Default::default(),
                }],
                response_id: None,
                finish_reason: Some(FinishReason::ToolUse),
                usage: None,
            },
            ChatResponse {
                messages: vec![Message::assistant("done")],
                response_id: None,
                finish_reason: Some(FinishReason::Stop),
                usage: None,
            },
        ];

        let tool = tool_fn(
            ToolDefinition::new(
                "echo",
                "Echo input",
                serde_json::json!({"type":"object","properties":{"msg":{"type":"string"}}}),
            )
            .unwrap(),
            |args| async move { Ok(serde_json::json!({ "echoed": args["msg"] })) },
        );

        let mut pipeline = MiddlewarePipeline::new();
        pipeline.add_function_middleware(RecordingFunctionMiddleware {
            label: "outer",
            log: log.clone(),
        });
        pipeline.add_function_middleware(RecordingFunctionMiddleware {
            label: "inner",
            log: log.clone(),
        });

        let agent = ChatClientAgent::builder()
            .client(ScriptedClient {
                responses: Arc::new(Mutex::new(responses)),
            })
            .tool(tool)
            .middleware(pipeline)
            .build()
            .unwrap();

        let mut session = AgentSession::new();
        let response = agent.run(vec![Message::user("go")], &mut session, None).await.unwrap();
        assert_eq!(response.text, "done");

        let log = log.lock().unwrap();
        // Outer runs first, returns last. Inner runs between. One invocation cycle.
        assert_eq!(*log, vec!["outer:before", "inner:before", "inner:after", "outer:after"]);
    }

    /// Agent middleware that records when it runs.
    struct RecordingAgentMiddleware {
        label: &'static str,
        log: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl AgentMiddleware for RecordingAgentMiddleware {
        async fn on_run(
            &self,
            messages: Vec<Message>,
            session: &mut AgentSession,
            next: &dyn AgentMiddlewareNext,
        ) -> AgentResult<AgentResponse> {
            self.log.lock().unwrap().push(format!("{}:enter", self.label));
            let out = next.run(messages, session).await;
            self.log.lock().unwrap().push(format!("{}:exit", self.label));
            out
        }
    }

    #[tokio::test]
    async fn agent_middleware_wraps_run_in_order() {
        let log = Arc::new(Mutex::new(Vec::<String>::new()));
        let response = ChatResponse {
            messages: vec![Message::assistant("ok")],
            response_id: None,
            finish_reason: Some(FinishReason::Stop),
            usage: None,
        };

        let mut pipeline = MiddlewarePipeline::new();
        pipeline.add_agent_middleware(RecordingAgentMiddleware {
            label: "A",
            log: log.clone(),
        });
        pipeline.add_agent_middleware(RecordingAgentMiddleware {
            label: "B",
            log: log.clone(),
        });

        let agent = ChatClientAgent::builder()
            .client(ScriptedClient {
                responses: Arc::new(Mutex::new(vec![response])),
            })
            .middleware(pipeline)
            .build()
            .unwrap();

        let mut session = AgentSession::new();
        agent.run(vec![Message::user("hi")], &mut session, None).await.unwrap();

        let log = log.lock().unwrap();
        assert_eq!(*log, vec!["A:enter", "B:enter", "B:exit", "A:exit"]);
    }

    /// Chat-client middleware that rewrites a user message.
    struct PrefixingChatMiddleware;
    #[async_trait]
    impl ChatClientMiddleware for PrefixingChatMiddleware {
        async fn on_get_response(
            &self,
            messages: &mut Vec<Message>,
            options: &mut ChatOptions,
            next: &dyn ChatClient,
        ) -> AgentResult<ChatResponse> {
            // Prepend "[prefixed] " to the first user message's text content.
            for m in messages.iter_mut() {
                if m.role == Role::User {
                    for c in m.content.iter_mut() {
                        if let Content::Text { text } = c {
                            *text = format!("[prefixed] {text}");
                        }
                    }
                }
            }
            next.get_response(messages, Some(options)).await
        }
    }

    /// Chat client that records what it actually sees.
    struct CapturingClient {
        captured: Arc<Mutex<Vec<Message>>>,
        response: ChatResponse,
    }

    #[async_trait]
    impl ChatClient for CapturingClient {
        async fn get_response(
            &self,
            messages: &[Message],
            _options: Option<&ChatOptions>,
        ) -> AgentResult<ChatResponse> {
            *self.captured.lock().unwrap() = messages.to_vec();
            Ok(self.response.clone())
        }
        async fn get_response_stream(
            &self,
            _: &[Message],
            _: Option<&ChatOptions>,
        ) -> AgentResult<crate::streaming::ResponseStream> {
            Err(AgentError::Unimplemented("stream".into()))
        }
    }

    #[tokio::test]
    async fn chat_client_middleware_modifies_request_before_it_reaches_inner() {
        let captured = Arc::new(Mutex::new(Vec::<Message>::new()));
        let response = ChatResponse {
            messages: vec![Message::assistant("fine")],
            response_id: None,
            finish_reason: Some(FinishReason::Stop),
            usage: None,
        };

        let mut pipeline = MiddlewarePipeline::new();
        pipeline.add_chat_client_middleware(PrefixingChatMiddleware);

        let agent = ChatClientAgent::builder()
            .client(CapturingClient {
                captured: captured.clone(),
                response,
            })
            .middleware(pipeline)
            .build()
            .unwrap();

        let mut session = AgentSession::new();
        agent
            .run(vec![Message::user("original")], &mut session, None)
            .await
            .unwrap();

        let seen = captured.lock().unwrap();
        // The inner client should see the prefixed message.
        let user_texts: Vec<String> = seen.iter().filter(|m| m.role == Role::User).map(|m| m.text()).collect();
        assert!(
            user_texts.iter().any(|t| t.starts_with("[prefixed] original")),
            "middleware should have rewritten the user message; saw: {user_texts:?}"
        );
    }

    // ---------- Streaming tests ----------

    use crate::streaming::ResponseStream;
    use crate::types::{ChatResponseUpdate, ToolCallUpdate};

    /// Streaming client that replays scripted `ChatResponseUpdate` sequences.
    struct ScriptedStreamClient {
        // One Vec<updates> per call to get_response_stream.
        scripts: Arc<Mutex<Vec<Vec<ChatResponseUpdate>>>>,
    }

    #[async_trait]
    impl ChatClient for ScriptedStreamClient {
        async fn get_response(&self, _: &[Message], _: Option<&ChatOptions>) -> AgentResult<ChatResponse> {
            Err(AgentError::Unimplemented("non-stream".into()))
        }

        async fn get_response_stream(&self, _: &[Message], _: Option<&ChatOptions>) -> AgentResult<ResponseStream> {
            let mut scripts = self.scripts.lock().unwrap();
            if scripts.is_empty() {
                return Err(AgentError::InvalidResponse("no scripted stream left".into()));
            }
            let updates = scripts.remove(0);
            let stream = async_stream::try_stream! {
                for u in updates {
                    yield u;
                }
            };
            Ok(ResponseStream::new(stream))
        }
    }

    #[tokio::test]
    async fn run_stream_yields_text_and_runs_tool_loop() {
        // Scripted streams:
        //   1. emit "let me check " text + tool_call "echo({msg:\"hi\"})" + finish_reason=ToolUse
        //   2. emit "result: hi" text + finish_reason=Stop
        let round1 = vec![
            ChatResponseUpdate {
                text: Some("let me check ".into()),
                tool_call: None,
                finish_reason: None,
                usage: None,
            },
            ChatResponseUpdate {
                text: None,
                tool_call: Some(ToolCallUpdate {
                    id: "call_1".into(),
                    name: "echo".into(),
                    arguments_delta: "{\"msg\":\"hi\"}".into(),
                }),
                finish_reason: None,
                usage: None,
            },
            ChatResponseUpdate {
                text: None,
                tool_call: None,
                finish_reason: Some(FinishReason::ToolUse),
                usage: None,
            },
        ];
        let round2 = vec![
            ChatResponseUpdate {
                text: Some("result: hi".into()),
                tool_call: None,
                finish_reason: None,
                usage: None,
            },
            ChatResponseUpdate {
                text: None,
                tool_call: None,
                finish_reason: Some(FinishReason::Stop),
                usage: None,
            },
        ];

        let tool = tool_fn(
            ToolDefinition::new(
                "echo",
                "Echo",
                serde_json::json!({"type":"object","properties":{"msg":{"type":"string"}}}),
            )
            .unwrap(),
            |args| async move { Ok(serde_json::json!({ "echoed": args["msg"] })) },
        );

        let agent = ChatClientAgent::builder()
            .client(ScriptedStreamClient {
                scripts: Arc::new(Mutex::new(vec![round1, round2])),
            })
            .tool(tool)
            .build()
            .unwrap();

        let mut session = AgentSession::new();
        let stream = agent.run_stream(vec![Message::user("go")], &mut session, None).unwrap();
        let text = stream.collect_text().await.unwrap();
        assert!(text.contains("let me check"));
        assert!(text.contains("result: hi"));
    }

    // ---------- Per-call options tests ----------

    /// Client that captures the ChatOptions it received, so tests can assert
    /// that per-call overrides flow through correctly.
    struct OptionsCapturingClient {
        captured: Arc<Mutex<Option<ChatOptions>>>,
        response: ChatResponse,
    }

    #[async_trait]
    impl ChatClient for OptionsCapturingClient {
        async fn get_response(
            &self,
            _messages: &[Message],
            options: Option<&ChatOptions>,
        ) -> AgentResult<ChatResponse> {
            *self.captured.lock().unwrap() = options.cloned();
            Ok(self.response.clone())
        }
        async fn get_response_stream(
            &self,
            _: &[Message],
            _: Option<&ChatOptions>,
        ) -> AgentResult<crate::streaming::ResponseStream> {
            Err(AgentError::Unimplemented("stream".into()))
        }
    }

    #[tokio::test]
    async fn per_call_chat_options_override_agent_defaults() {
        use crate::types::AgentRunOptions;

        let captured = Arc::new(Mutex::new(None));
        let response = ChatResponse {
            messages: vec![Message::assistant("ok")],
            response_id: None,
            finish_reason: Some(FinishReason::Stop),
            usage: None,
        };

        let defaults = ChatOptions {
            temperature: Some(0.2),
            max_tokens: Some(100),
            ..Default::default()
        };

        let agent = ChatClientAgent::builder()
            .client(OptionsCapturingClient {
                captured: captured.clone(),
                response,
            })
            .options(defaults)
            .build()
            .unwrap();

        // Per-call overrides: raise temperature, leave max_tokens alone.
        let overrides = ChatOptions {
            temperature: Some(0.9),
            ..Default::default()
        };
        let run_opts = AgentRunOptions::new().with_chat_options(overrides);

        let mut session = AgentSession::new();
        agent
            .run(vec![Message::user("hi")], &mut session, Some(&run_opts))
            .await
            .unwrap();

        let seen = captured.lock().unwrap().clone().expect("client saw options");
        assert_eq!(seen.temperature, Some(0.9), "override should replace default");
        assert_eq!(seen.max_tokens, Some(100), "untouched field should be preserved");
    }

    #[tokio::test]
    async fn per_call_additional_instructions_are_prepended() {
        let captured = Arc::new(Mutex::new(Vec::<Message>::new()));
        let response = ChatResponse {
            messages: vec![Message::assistant("ok")],
            response_id: None,
            finish_reason: Some(FinishReason::Stop),
            usage: None,
        };

        let agent = ChatClientAgent::builder()
            .client(CapturingClient {
                captured: captured.clone(),
                response,
            })
            .instructions("be concise")
            .build()
            .unwrap();

        use crate::types::AgentRunOptions;
        let run_opts = AgentRunOptions::new().with_additional_instructions("also: be cheerful");

        let mut session = AgentSession::new();
        agent
            .run(vec![Message::user("hi")], &mut session, Some(&run_opts))
            .await
            .unwrap();

        let seen = captured.lock().unwrap();
        let system_texts: Vec<String> = seen
            .iter()
            .filter(|m| m.role == Role::System)
            .map(|m| m.text())
            .collect();
        assert_eq!(system_texts.len(), 2, "both default + additional instructions present");
        assert_eq!(system_texts[0], "be concise");
        assert_eq!(system_texts[1], "also: be cheerful");
    }
}
