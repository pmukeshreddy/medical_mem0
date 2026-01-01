"""LLM orchestrator for chat and summarization."""

from typing import List, Dict
from openai import OpenAI
from config import get_settings


SYSTEM_PROMPT = """You are a medical assistant with access to patient memory.
Use the provided patient history to answer questions accurately.
If information is not in the patient's records, say so clearly.
Be concise and professional."""


class LLMOrchestrator:
    """Orchestrates LLM calls with memory context."""
    
    def __init__(self):
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
    
    def chat(
        self, 
        message: str, 
        memories: List[Dict],
        history: List[Dict] = None
    ) -> str:
        """
        Generate chat response with memory context.
        
        Args:
            message: User message
            memories: Retrieved memories for context
            history: Previous chat messages
        
        Returns:
            Assistant response
        """
        # Build context from memories
        context_parts = []
        for m in memories:
            content = m.get("content", m.get("memory", ""))
            if content:
                context_parts.append(f"- {content}")
        
        context = "\n".join(context_parts) if context_parts else "No relevant history found."
        
        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Patient History:\n{context}"}
        ]
        
        # Add chat history
        if history:
            for h in history[-10:]:  # Last 10 messages
                messages.append({"role": h["role"], "content": h["content"]})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Call LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def summarize_visit(self, visit_text: str) -> str:
        """Summarize a visit note."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Summarize this medical visit note concisely."},
                {"role": "user", "content": visit_text}
            ],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system", 
                    "content": """Extract medical entities from text. Return JSON:
{"conditions": [], "medications": [], "symptoms": [], "vitals": []}"""
                },
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=300
        )
        
        import json
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"conditions": [], "medications": [], "symptoms": [], "vitals": []}


# Singleton
_llm_orchestrator = None


def get_llm_orchestrator() -> LLMOrchestrator:
    """Get or create LLM orchestrator singleton."""
    global _llm_orchestrator
    if _llm_orchestrator is None:
        _llm_orchestrator = LLMOrchestrator()
    return _llm_orchestrator
