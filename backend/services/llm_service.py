"""
services/llm_service.py — EL CEREBRO DE LA APLICACIÓN (IA).
---------------------------------------------------------
Ahora simplificada exclusivamente para Gemini y evitar dependencias innecesarias 
en entornos de bajos recursos.
"""

import os
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  INTERFAZ ABSTRACTA
# ═══════════════════════════════════════════════════════════════

class BaseLLMProvider(ABC):
    """Contrato base que deben implementar todos los proveedores de IA."""

    @abstractmethod
    def summarize(self, text: str) -> str:
        """Sintetiza el contenido de un documento en unos pocos párrafos clave."""
        ...

    @abstractmethod
    def chat(self, prompt: str, context: str, history: list[dict] = None) -> str:
        """Genera una respuesta sopesando la pregunta contra el contexto corporativo."""
        ...

    @abstractmethod
    def chat_stream(self, prompt: str, context: str, history: list[dict] = None):
        """Versión asíncrona por generador para interfaces de chat fluido."""
        ...


# ═══════════════════════════════════════════════════════════════
#  PROVEEDOR GOOGLE GEMINI
# ═══════════════════════════════════════════════════════════════

class GeminiProvider(BaseLLMProvider):
    """
    Integración con Google Gemini (Generative AI).

    Especialmente útil por su gran ventana de contexto y velocidad en 
    tareas de razonamiento intermedio. Requiere GEMINI_API_KEY.
    """
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model_name = os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash")
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"✅ Gemini LLM inicializado: {model_name}")

    def summarize(self, text: str) -> str:
        prompt = (
            f"Resume el siguiente texto corporativo en 3-5 frases concisas en español. "
            f"Responde SOLO con el resumen:\n\n{text}"
        )
        return self.model.generate_content(prompt).text.strip()

    def _build_content_gemini(self, prompt: str, context: str, history: list[dict] = None) -> str:
        historico = ""
        if history:
            for msg in history[-6:]:
                role = "Usuario" if msg.get("role") == "user" else "Asistente"
                historico += f"{role}: {msg.get('content')}\n\n"
        
        full_prompt = (
            f"Eres un asistente corporativo. Responde basándote SOLO en el contexto. "
            f"Si la información no está en el documento, indícalo claramente.\n"
            f"Usa notación [1], [2] para citar las fuentes numeradas.\n\n"
            f"Contexto:\n{context}\n\n"
        )
        if historico:
            full_prompt += f"Historial:\n{historico}"
            
        full_prompt += f"Pregunta: {prompt}"
        return full_prompt

    def chat(self, prompt: str, context: str, history: list[dict] = None) -> str:
        full_prompt = self._build_content_gemini(prompt, context, history)
        return self.model.generate_content(full_prompt).text.strip()

    def chat_stream(self, prompt: str, context: str, history: list[dict] = None):
        full_prompt = self._build_content_gemini(prompt, context, history)
        response = self.model.generate_content(full_prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text


# ═══════════════════════════════════════════════════════════════
#  FACTORY — selecciona el proveedor (forzado a Gemini)
# ═══════════════════════════════════════════════════════════════

class LLMFactory:
    _instance: BaseLLMProvider | None = None

    @classmethod
    def get_provider(cls) -> BaseLLMProvider:
        if cls._instance is None:
            logger.info(f"🏭 Inicializando LLM provider forzado a: Gemini")
            cls._instance = GeminiProvider()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None


def get_llm_service() -> BaseLLMProvider:
    return LLMFactory.get_provider()
