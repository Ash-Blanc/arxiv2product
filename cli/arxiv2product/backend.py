import os
from agno.models.openai import OpenAIChat
from agno.models.openrouter import OpenRouter

OPENAI_COMPATIBLE_BACKEND = "openai_compatible"
DEFAULT_OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_DIRECT_TIMEOUT_SECONDS = 240.0

def get_execution_backend_name() -> str:
    configured = os.getenv("EXECUTION_BACKEND", "").strip().lower()
    if configured in {OPENAI_COMPATIBLE_BACKEND, "agentica"}:
        return configured
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"):
        return OPENAI_COMPATIBLE_BACKEND
    return "agentica"

def normalize_model_name(model: str) -> str:
    return model.removeprefix("openrouter:")

def get_agno_model(model_id: str):
    """Returns an Agno model instance based on configuration."""
    backend = get_execution_backend_name()
    model_name = normalize_model_name(model_id)
    
    # If the user explicitly provided an OpenAI key but NO OpenRouter key, use OpenAI.
    if os.getenv("OPENAI_API_KEY") and not os.getenv("OPENROUTER_API_KEY"):
        return OpenAIChat(id=model_name)
    
    # Otherwise default to OpenRouter
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    
    return OpenRouter(
        id=model_name,
        api_key=api_key,
        base_url=base_url,
    )

