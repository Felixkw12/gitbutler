use anyhow::Result;
use async_openai::{Client, config::OpenAIConfig};
use but_secret::{Sensitive, secret};
use but_tools::tool::Toolset;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::{
    AI_OPPER_SECRET_HANDLE,
    chat::ChatMessage,
    client::LLMClient,
    openai_utils::{
        OpenAIClientProvider, response_blocking, stream_response_blocking,
        structured_output_blocking, tool_calling_loop, tool_calling_loop_stream,
    },
};

const OPPER_API_BASE_DEFAULT: &str = "https://api.opper.ai/v3/compat";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OpperConfig {
    pub api_base: String,
}

impl Default for OpperConfig {
    fn default() -> Self {
        Self {
            api_base: OPPER_API_BASE_DEFAULT.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OpperProvider {
    model: Option<String>,
    config: OpperConfig,
    api_key: Sensitive<String>,
}

impl OpperProvider {
    pub fn with(config: Option<OpperConfig>, model: Option<String>) -> Option<Self> {
        let config = config.unwrap_or_default();
        let api_key = Self::retrieve_api_key()?;
        Some(Self {
            config,
            model,
            api_key,
        })
    }

    fn retrieve_api_key() -> Option<Sensitive<String>> {
        // Try secret storage first, then fall back to env var
        if let Ok(Some(key)) = secret::retrieve(AI_OPPER_SECRET_HANDLE, secret::Namespace::Global)
            && !key.0.trim().is_empty()
        {
            return Some(key);
        }
        if let Ok(val) = std::env::var("OPPER_API_KEY")
            && !val.trim().is_empty()
        {
            return Some(Sensitive(val));
        }
        None
    }
}

impl OpenAIClientProvider for OpperProvider {
    fn client(&self) -> Result<Client<OpenAIConfig>> {
        let open_ai_config = OpenAIConfig::new()
            .with_api_base(self.config.api_base.clone())
            .with_api_key(self.api_key.0.clone());

        Ok(Client::with_config(open_ai_config))
    }
}

impl LLMClient for OpperProvider {
    fn model(&self) -> Option<String> {
        self.model.clone()
    }

    fn tool_calling_loop_stream(
        &self,
        system_message: &str,
        chat_messages: Vec<ChatMessage>,
        tool_set: &mut impl Toolset,
        model: &str,
        on_token: impl Fn(&str) + Send + Sync + 'static,
    ) -> Result<(String, Vec<ChatMessage>)> {
        let result = tool_calling_loop_stream(
            self,
            system_message,
            chat_messages,
            tool_set,
            model,
            on_token,
        )?;
        Ok((result.final_response, result.message_history))
    }

    fn tool_calling_loop(
        &self,
        system_message: &str,
        chat_messages: Vec<ChatMessage>,
        tool_set: &mut impl Toolset,
        model: &str,
    ) -> Result<String> {
        tool_calling_loop(self, system_message, chat_messages, tool_set, model)
    }

    fn stream_response(
        &self,
        system_message: &str,
        chat_messages: Vec<ChatMessage>,
        model: &str,
        on_token: impl Fn(&str) + Send + Sync + 'static,
    ) -> Result<Option<String>> {
        stream_response_blocking(self, system_message, chat_messages, model, on_token)
    }

    fn structured_output<
        T: serde::Serialize + DeserializeOwned + JsonSchema + std::marker::Send + 'static,
    >(
        &self,
        system_message: &str,
        chat_messages: Vec<ChatMessage>,
        model: &str,
    ) -> Result<Option<T>> {
        structured_output_blocking::<T>(self, system_message, chat_messages, model)
    }

    fn response(
        &self,
        system_message: &str,
        chat_messages: Vec<ChatMessage>,
        model: &str,
    ) -> Result<Option<String>> {
        response_blocking(self, system_message, chat_messages, model)
    }
}
