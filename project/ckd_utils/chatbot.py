# ckd_utils/chatbot.py
# ─────────────────────────────────────────────────────────────────────────────
# CKD Medical Assistant Chatbot
#
# Architecture (hybrid approach):
#   1. Offline Knowledge Base search (primary, always available, < 2s)
#   2. Google Gemini API fallback (if GEMINI_API_KEY env var is set)
#   3. OpenAI API fallback       (if OPENAI_API_KEY env var is set)
#   4. Graceful "I don't know" message if neither key is available
#
# Safety:
#   - Always appends a medical disclaimer to LLM-generated responses.
#   - Never suggests diagnoses or prescribes medications.
#   - Keeps responses under 300 words.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import re
import time
import logging
from typing import Optional
from dotenv import load_dotenv

from ckd_utils.knowledge_base import get_knowledge_base

# Ensure we load the .env from the correct root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(BASE_DIR, ".env")
dotenv_loaded = load_dotenv(env_path)

logger = logging.getLogger(__name__)

if dotenv_loaded:
    logger.info(f".env loaded successfully from {env_path}")
else:
    logger.warning(f"Could not load .env file from {env_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MEDICAL_DISCLAIMER = (
    "\n\n---\n"
    "⚠️ *This information is **educational only** and is **not a substitute** for "
    "professional medical advice. Always consult a qualified healthcare provider "
    "for medical decisions.*"
)

# System prompt used when falling back to an LLM API
LLM_SYSTEM_PROMPT = """You are a knowledgeable and empathetic CKD (Chronic Kidney Disease) 
Medical Assistant chatbot embedded in a clinical decision-support tool. Your role is to:

1. Explain CKD stages, medical terms, and test results in simple, clear language.
2. Provide general kidney health information and recommendations.
3. Answer questions about the prediction application.
4. Guide users on lifestyle, diet, and exercise for kidney health.

CRITICAL RULES:
- NEVER diagnose diseases or prescribe medications.
- NEVER claim to replace a doctor or nephrologist.
- ALWAYS recommend consulting a healthcare professional for medical decisions.
- Keep responses under 300 words.
- Use markdown formatting for clarity (bullet points, bold text).
- Be compassionate and reassuring, especially for patients with advanced CKD.
- If asked something outside CKD/kidney health, politely redirect.

You are NOT a general chatbot. Stay focused on kidney health and CKD.
"""

# Suggested quick questions shown in the UI
QUICK_QUESTIONS = [
    "What is CKD?",
    "Explain my prediction",
    "What is eGFR?",
    "What foods should I avoid?",
    "How can I protect my kidneys?",
    "What are the symptoms of CKD?",
    "What is the difference between CKD stages?",
]

# Fallback message when no answer is found and no LLM is configured
FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't find a specific answer to that question in my knowledge base.\n\n"
    "Here are some things I **can** help you with:\n"
    "- Explaining CKD stages and what they mean\n"
    "- Defining medical terms (eGFR, creatinine, BUN, etc.)\n"
    "- Providing kidney health tips and diet recommendations\n"
    "- Explaining your prediction result\n"
    "- Answering common CKD questions\n\n"
    "Try asking one of the quick questions above, or rephrasing your question.\n\n"
    "For specific medical advice, please consult your **nephrologist** or "
    "**primary care physician**."
)

GREETING_MESSAGE = (
    "👋 Hello! I'm your **Chronic Kidney Disease Medical Assistant**.\n\n"
    "I can help you:\n"
    "- 🏥 Understand Chronic Kidney Disease stages and medical terms\n"
    "- 📊 Explain your prediction results\n"
    "- 🥗 Learn about kidney-friendly diet and lifestyle\n"
    "- 💊 Understand medications and tests\n"
    "- 🗺️ Navigate this application\n\n"
    "Use the quick question buttons above or type your question below.\n\n"
    "⚠️ *I provide educational information only — not medical advice.*"
)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Availability Checks (lazy imports with graceful failure)
# ─────────────────────────────────────────────────────────────────────────────

def _get_clean_error_message(exception: Exception) -> str:
    err_str = str(exception)
    if "403" in err_str or "API_KEY_INVALID" in err_str or "invalid" in err_str.lower():
        return "Invalid API Key"
    if "permission" in err_str.lower() or "denied" in err_str.lower():
        return "Permission Denied"
    if "429" in err_str or "quota" in err_str.lower() or "exhausted" in err_str.lower():
        return "Quota Exceeded"
    if "network" in err_str.lower() or "connection" in err_str.lower() or "dns" in err_str.lower():
        return "Network Error"
    return f"Gemini API Error: {err_str}"


def _get_gemini_api_key() -> str:
    """Retrieve GEMINI_API_KEY from environment or Streamlit secrets."""
    # Use os.getenv as requested
    key = os.getenv("GEMINI_API_KEY", "")
    if key:
        return key
    # Fall back to Streamlit secrets
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def _check_gemini_available() -> bool:
    """Check if Gemini API library is installed."""
    try:
        import google.generativeai  # noqa: F401
        return True
    except ImportError:
        return False


def _check_openai_available() -> bool:
    """Check if OpenAI API is available (env key + library installed)."""
    if not os.environ.get("OPENAI_API_KEY"):
        return False
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# LLM Response Functions
# ─────────────────────────────────────────────────────────────────────────────

def _query_gemini(user_message: str, model_name: str, context: Optional[str] = None) -> Optional[str]:
    """
    Query Google Gemini API for a response using google-generativeai.
    """
    try:
        import google.generativeai as genai

        api_key = _get_gemini_api_key()
        if not api_key:
            return None

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=LLM_SYSTEM_PROMPT,
        )

        # Build the full prompt with optional prediction context
        if context:
            user_prompt = (
                f"[Patient Prediction Context]\n{context}\n\n"
                f"[User Question]\n{user_message}"
            )
        else:
            user_prompt = user_message

        response = model.generate_content(
            user_prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=400,
                temperature=0.3,
            ),
        )

        if response and response.text:
            return _truncate_to_word_limit(response.text, max_words=300)

    except Exception as e:
        logger.warning(f"Gemini API error: {e}")
        raise  # Re-raise so the caller can surface a meaningful error
