"""
services/llm_service.py — Servicio LLM centralizado (Factory Pattern).

Permite cambiar de proveedor simplemente con la variable de entorno:
    LLM_PROVIDER=local    → HuggingFaceTB/SmolLM2-135M (CPU, sin coste)
    LLM_PROVIDER=openai   → OpenAI (requiere OPENAI_API_KEY)
    LLM_PROVIDER=gemini   → Google Gemini (requiere GEMINI_API_KEY)
    LLM_PROVIDER=claude   → Anthropic Claude (requiere ANTHROPIC_API_KEY)

Uso:
    from services.llm_service import get_llm_service
    llm = get_llm_service()
    summary = llm.summarize(text)
    answer  = llm.chat("¿Cuál es el presupuesto?", context=full_text)
"""

import os
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  INTERFAZ ABSTRACTA
# ═══════════════════════════════════════════════════════════════

class BaseLLMProvider(ABC):
    """Interfaz común para todos los proveedores LLM."""

    @abstractmethod
    def summarize(self, text: str) -> str:
        """Genera un resumen conciso del texto dado."""
        ...

    @abstractmethod
    def chat(self, prompt: str, context: str, history: list[dict] = None) -> str:
        """Responde una pregunta usando el contexto del documento y opcionalmente historial."""
        ...

    @abstractmethod
    def chat_stream(self, prompt: str, context: str, history: list[dict] = None):
        """Versión en streaming de chat(). Retorna un generador de strings."""
        ...

    @abstractmethod
    def expand_keywords(self, query: str) -> str:
        """Extrae palabras clave relevantes de una consulta de búsqueda."""
        ...


# ═══════════════════════════════════════════════════════════════
#  PROVEEDOR LOCAL — HuggingFaceTB/SmolLM2-135M (CPU)
# ═══════════════════════════════════════════════════════════════

class LocalSmolLMProvider(BaseLLMProvider):
    """
    Proveedor local usando SmolLM2-135M via transformers pipeline.
    Carga el modelo en la primera llamada (lazy loading) y lo reutiliza.
    Aplica cuantización dinámica int8 para reducir uso de RAM.
    """
    _pipeline = None

    def _get_pipeline(self):
        if LocalSmolLMProvider._pipeline is None:
            import torch
            from transformers import pipeline as hf_pipeline

            logger.info("🔄 Cargando HuggingFaceTB/SmolLM2-135M (LLM local)...")
            pipe = hf_pipeline(
                "text-generation",
                model="HuggingFaceTB/SmolLM2-135M",
                device=-1,          # CPU
                dtype=torch.float32,
            )
            # Cuantización int8 para reducir footprint de RAM
            pipe.model = torch.quantization.quantize_dynamic(
                pipe.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            pipe.model.eval()
            LocalSmolLMProvider._pipeline = pipe
            logger.info("✅ SmolLM2-135M cargado y cuantizado")

        return LocalSmolLMProvider._pipeline

    def _generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        pipe = self._get_pipeline()
        result = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.2,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        generated = result[0]["generated_text"]
        # Recortar el prompt del output
        return generated[len(prompt):].strip()

    def summarize(self, text: str) -> str:
        prompt = (
            f"Resume el siguiente texto corporativo en 3-5 frases en español:\n\n"
            f"{text[:2000]}\n\n"
            f"Resumen:"
        )
        return self._generate(prompt, max_new_tokens=200)

    def _format_history_smollm(self, context: str, prompt: str, history: list[dict] = None) -> str:
        historico = ""
        if history:
            for msg in history[-4:]: # Solo los 4 últimos para no saturar contexto
                role = "Humano" if msg.get("role") == "user" else "Asistente"
                historico += f"{role}: {msg.get('content')}\n\n"
        
        full_prompt = (
            f"Contexto del documento:\n{context[:3000]}\n\n"
        )
        if historico:
            full_prompt += f"Historial de conversación:\n{historico}"
        
        full_prompt += f"Pregunta: {prompt}\n\nRespuesta:"
        return full_prompt

    def chat(self, prompt: str, context: str, history: list[dict] = None) -> str:
        full_prompt = self._format_history_smollm(context, prompt, history)
        return self._generate(full_prompt, max_new_tokens=250)

    def chat_stream(self, prompt: str, context: str, history: list[dict] = None):
        # SmolLM/Transformers pipeline no tiene un streamer súper limpio out-of-the-box sin TextStreamer
        # Para simplificar y no bloquear, hacemos un fake stream (devolvemos todo de golpe)
        # En un entorno real se usaría TextIteratorStreamer de transformers
        ans = self.chat(prompt, context, history)
        # Dividimos en palabras para fingir el stream
        for word in ans.split(" "):
            yield word + " "

    def expand_keywords(self, query: str) -> str:
        import re
        pipe = self._get_pipeline()
        result = pipe(
            query,
            max_new_tokens=32,
            do_sample=True,
            temperature=0.75,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        generated = result[0]["generated_text"]
        response = generated[len(query):]
        first_line = response.split("\n")[0]
        tokens = re.findall(r"[\w'\-]+", first_line)
        clean = ", ".join(tokens) if tokens else ", ".join(query.split())
        logger.info(f"🧠 SmolLM keywords: '{query}' -> '{clean}'")
        return clean


# ═══════════════════════════════════════════════════════════════
#  PROVEEDOR OPENAI
# ═══════════════════════════════════════════════════════════════

class OpenAIProvider(BaseLLMProvider):
    """
    Proveedor OpenAI. Requiere:
        OPENAI_API_KEY
        OPENAI_LLM_MODEL (opcional, default: gpt-4o-mini)
    """
    def __init__(self):
        import openai
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
        logger.info(f"✅ OpenAI LLM inicializado: {self.model}")

    def _complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=512,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    def summarize(self, text: str) -> str:
        return self._complete(
            system=(
                "Eres un asistente experto en resumir documentos corporativos en español. "
                "Responde SOLO con el resumen, sin preámbulos ni explicaciones."
            ),
            user=f"Resume este texto en 3-5 frases:\n\n{text}",
        )

    def _build_messages_openai(self, prompt: str, context: str, history: list[dict] = None) -> list[dict]:
        system = (
            "Eres un asistente corporativo. Responde basándote EXCLUSIVAMENTE "
            "en el contexto proporcionado. Usa [1], [2], etc. para citar las fuentes "
            "si se te proporcionan identificadores. Si no está en el documento, indícalo."
        )
        msgs = [{"role": "system", "content": f"{system}\n\nContexto:\n{context}"}]
        
        if history:
            for h in history[-6:]: # max 6 mensajes de historial
                role = "assistant" if h.get("role") == "assistant" else "user"
                msgs.append({"role": role, "content": h.get("content", "")})
        
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def chat(self, prompt: str, context: str, history: list[dict] = None) -> str:
        msgs = self._build_messages_openai(prompt, context, history)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            max_tokens=512,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()

    def chat_stream(self, prompt: str, context: str, history: list[dict] = None):
        msgs = self._build_messages_openai(prompt, context, history)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            max_tokens=512,
            temperature=0.3,
            stream=True,
        )
        for chunk in resp:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def expand_keywords(self, query: str) -> str:
        return self._complete(
            system=(
                "Extract search keywords from the given query. "
                "Return ONLY a comma-separated list of relevant keywords, no explanations."
            ),
            user=f"Query: {query}\nKeywords:",
        )


# ═══════════════════════════════════════════════════════════════
#  PROVEEDOR GOOGLE GEMINI
# ═══════════════════════════════════════════════════════════════

class GeminiProvider(BaseLLMProvider):
    """
    Proveedor Google Gemini. Requiere:
        GEMINI_API_KEY
        GEMINI_LLM_MODEL (opcional, default: gemini-2.5-flash)
    """
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model_name = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")
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

    def expand_keywords(self, query: str) -> str:
        prompt = (
            f"Concepts fitting to this definition: {query}\n"
        )
        return self.model.generate_content(prompt).text.strip()


# ═══════════════════════════════════════════════════════════════
#  PROVEEDOR ANTHROPIC CLAUDE
# ═══════════════════════════════════════════════════════════════

class ClaudeProvider(BaseLLMProvider):
    """
    Proveedor Anthropic Claude. Requiere:
        ANTHROPIC_API_KEY
        CLAUDE_LLM_MODEL (opcional, default: claude-3-haiku-20240307)
    """
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("CLAUDE_LLM_MODEL", "claude-3-haiku-20240307")
        logger.info(f"✅ Claude LLM inicializado: {self.model}")

    def _complete(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text.strip()

    def summarize(self, text: str) -> str:
        return self._complete(
            system=(
                "Eres un asistente experto en resumir documentos corporativos. "
                "Responde SOLO con el resumen en español, sin preámbulos."
            ),
            user=f"Resume este texto en 3-5 frases:\n\n{text}",
        )

    def _build_messages_claude(self, prompt: str, context: str, history: list[dict] = None):
        system = (
            "Eres un asistente corporativo. Responde basándote EXCLUSIVAMENTE "
            "en el contexto. Usa identificadores [1], [2] para citar fuentes. "
            "Si la respuesta no está en el documento, indícalo claramente.\n\n"
            f"Contexto:\n{context}"
        )
        msgs = []
        if history:
            for h in history[-6:]:
                role = "assistant" if h.get("role") == "assistant" else "user"
                msgs.append({"role": role, "content": h.get("content", "")})
        
        msgs.append({"role": "user", "content": prompt})
        return system, msgs

    def chat(self, prompt: str, context: str, history: list[dict] = None) -> str:
        system, msgs = self._build_messages_claude(prompt, context, history)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system,
            messages=msgs,
        )
        return resp.content[0].text.strip()

    def chat_stream(self, prompt: str, context: str, history: list[dict] = None):
        system, msgs = self._build_messages_claude(prompt, context, history)
        with self.client.messages.stream(
            model=self.model,
            max_tokens=512,
            system=system,
            messages=msgs,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def expand_keywords(self, query: str) -> str:
        return self._complete(
            system=(
                "Extract search keywords from the given query. "
                "Return ONLY a comma-separated list of relevant keywords, no explanations."
            ),
            user=f"Query: {query}\nKeywords:",
        )


# ═══════════════════════════════════════════════════════════════
#  FACTORY — selecciona el proveedor por variable de entorno
# ═══════════════════════════════════════════════════════════════

class LLMFactory:
    """
    Singleton factory. Lee LLM_PROVIDER del entorno y devuelve
    la instancia adecuada. La instancia se crea una sola vez.
    """
    _instance: BaseLLMProvider | None = None

    @classmethod
    def get_provider(cls) -> BaseLLMProvider:
        if cls._instance is None:
            provider = os.getenv("LLM_PROVIDER", "local").lower()
            logger.info(f"🏭 Inicializando LLM provider: '{provider}'")
            if provider == "openai":
                cls._instance = OpenAIProvider()
            elif provider == "gemini":
                cls._instance = GeminiProvider()
            elif provider == "claude":
                cls._instance = ClaudeProvider()
            else:
                # Fallback a local si el valor no es reconocido
                if provider not in ("local",):
                    logger.warning(f"⚠️  LLM_PROVIDER='{provider}' no reconocido, usando 'local'")
                cls._instance = LocalSmolLMProvider()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Invalida la instancia en caché para forzar re-inicialización con nuevo proveedor."""
        cls._instance = None


def get_llm_service() -> BaseLLMProvider:
    """Acceso global al proveedor LLM activo. Importar desde cualquier módulo."""
    return LLMFactory.get_provider()
