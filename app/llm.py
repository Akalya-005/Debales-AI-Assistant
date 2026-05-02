"""Centralized LLM provider selection for the Debales AI assistant."""

from typing import Any

from app.config import Settings

def get_chat_model(settings: Settings) -> Any | None:
    provider = settings.chat_provider
    if provider == "openai":
        if not settings.openai_api_key:
            return None
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_chat_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )
    elif provider == "groq":
        if not settings.groq_api_key:
            return None
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.1-8b-instant", # or another groq model
            temperature=0,
            api_key=settings.groq_api_key,
        )
    elif provider == "gemini":
        if not settings.google_api_key:
            return None
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=settings.google_api_key,
        )
    else:
        raise ValueError(f"Unknown CHAT_PROVIDER: {provider}")
