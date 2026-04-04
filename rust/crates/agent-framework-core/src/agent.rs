// Copyright (c) Microsoft. All rights reserved.

use async_trait::async_trait;
use tracing::{debug, instrument};

use crate::client::ChatClient;
use crate::context::ContextProvider;
use crate::error::{AgentError, AgentResult};
use crate::middleware::MiddlewarePipeline;
use crate::session::AgentSession;
use crate::streaming::AgentResponseStream;
use crate::tools::{FunctionTool, ToolDefinition};
use crate::types::{AgentResponse, ChatOptions, Content, FinishReason, Message, Role};

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
    /// This is the primary entry point. The agent will invoke the underlying
    /// model, execute any tool calls, and return the final response.
    async fn run(&self, messages: Vec<Message>, session: &mut AgentSession) -> AgentResult<AgentResponse>;

    /// Run the agent and return a stream of incremental updates.
    fn run_stream(&self, messages: Vec<Message>, session: &mut AgentSession) -> AgentResult<AgentResponseStream>;
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
/// let response = agent.run(vec![Message::user("Hello!")], &mut session).await.unwrap();
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
    #[allow(dead_code)]
    middleware: MiddlewarePipeline,
}

impl ChatClientAgent {
    /// Create a builder for constructing a `ChatClientAgent`.
    pub fn builder() -> ChatClientAgentBuilder {
        ChatClientAgentBuilder::default()
    }

    /// Get the tool definitions for all registered tools.
    fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|t| t.definition().clone()).collect()
    }

    /// Find a tool by name.
    fn find_tool(&self, name: &str) -> Option<&dyn FunctionTool> {
        self.tools
            .iter()
            .find(|t| t.definition().name == name)
            .map(|t| t.as_ref())
    }

    /// Build the full message list including system instructions and context.
    async fn build_messages(&self, input_messages: &[Message], session: &AgentSession) -> AgentResult<Vec<Message>> {
        let mut messages = Vec::new();

        // Add system instructions.
        if let Some(instructions) = &self.instructions {
            messages.push(Message::system(instructions.clone()));
        }

        // Add context from providers.
        for provider in &self.context_providers {
            let context = provider.provide_context(session).await?;
            if let Some(instructions) = context.instructions {
                messages.push(Message::system(instructions));
            }
            messages.extend(context.messages);
        }

        // Add conversation history.
        let history = session.get_history().await?;
        messages.extend(history);

        // Add input messages.
        messages.extend(input_messages.iter().cloned());

        Ok(messages)
    }

    /// Execute the tool-call loop: call model, handle tool calls, repeat.
    #[instrument(skip(self, session), fields(agent_id = %self.id))]
    async fn execute_run(
        &self,
        input_messages: Vec<Message>,
        session: &mut AgentSession,
    ) -> AgentResult<AgentResponse> {
        let mut all_messages = self.build_messages(&input_messages, session).await?;
        let mut accumulated_messages = Vec::new();
        let tool_defs = self.tool_definitions();

        let mut options = self.options.clone();
        // Set tool definitions in extra if we have tools.
        if !tool_defs.is_empty() {
            options.extra.insert(
                "_tools".to_string(),
                serde_json::to_value(&tool_defs).unwrap_or_default(),
            );
        }

        for round in 0..self.max_tool_rounds {
            debug!(round, "Invoking chat client");

            let response = self.client.get_response(&all_messages, Some(&options)).await?;

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
                // No tool calls or model is done — return the response.
                // Save to history.
                let mut history_messages = input_messages.clone();
                history_messages.extend(accumulated_messages.clone());
                session.save_history(&history_messages).await?;

                return Ok(AgentResponse::from_chat_response(response, accumulated_messages));
            }

            // Add assistant messages (with tool calls) to the conversation.
            all_messages.extend(response.messages.clone());

            // Execute tool calls.
            for (id, name, args) in tool_calls {
                debug!(tool = %name, "Invoking tool");
                let result = match self.find_tool(&name) {
                    Some(tool) => match tool.invoke(args).await {
                        Ok(value) => serde_json::to_string(&value).unwrap_or_default(),
                        Err(e) => format!("Error: {e}"),
                    },
                    None => format!("Error: unknown tool '{name}'"),
                };

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

    async fn run(&self, messages: Vec<Message>, session: &mut AgentSession) -> AgentResult<AgentResponse> {
        self.execute_run(messages, session).await
    }

    fn run_stream(&self, _messages: Vec<Message>, _session: &mut AgentSession) -> AgentResult<AgentResponseStream> {
        // Streaming support will be fully implemented in a future iteration.
        // For now, return an error indicating it's not yet available.
        Err(AgentError::InvalidRequest(
            "Streaming not yet implemented for ChatClientAgent".to_string(),
        ))
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

        Ok(ChatClientAgent {
            id: uuid::Uuid::new_v4().to_string(),
            name: self.name,
            description: self.description,
            instructions: self.instructions,
            client,
            tools: self.tools,
            context_providers: self.context_providers,
            options: self.options,
            max_tool_rounds: self.max_tool_rounds.unwrap_or(10),
            middleware: self.middleware.unwrap_or_default(),
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

        fn get_response_stream(
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
        let response = agent.run(vec![Message::user("Hi")], &mut session).await.unwrap();

        assert_eq!(response.text, "Hello! How can I help you?");
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    }

    #[tokio::test]
    async fn test_builder_requires_client() {
        let result = ChatClientAgent::builder().name("NoClient").build();
        assert!(result.is_err());
    }
}
