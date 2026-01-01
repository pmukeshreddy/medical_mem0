"""
HyDE: Hypothetical Document Embeddings

Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022)
https://arxiv.org/abs/2212.10496

Key Idea:
- Questions and answers live in different embedding spaces
- Generate a hypothetical answer first, then use THAT for retrieval
- Hypothetical answer is closer to real documents than the question

Flow:
    Normal:  Query → Embed(Query) → Search
    HyDE:    Query → LLM(Generate Hypothetical) → Embed(Hypothetical) → Search

Example:
    Query: "What's the patient's diabetes status?"
    
    Hypothetical: "Patient has Type 2 Diabetes diagnosed in 2019,
                   currently managed with Metformin 1000mg twice daily.
                   Most recent A1C was 7.2% from January 2024."
    
    Search uses embedding of hypothetical → Better matches
"""

import os
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from mem0 import Memory
from openai import OpenAI


# Prompt templates for different query types
HYDE_PROMPTS = {
    "medical_history": """You are a medical records system. Given a question about a patient, 
generate a realistic hypothetical medical record entry that would answer this question.
Be specific with dates, medications, dosages, and values. Keep it concise (2-3 sentences).

Question: {query}

Hypothetical medical record entry:""",

    "condition": """Generate a realistic clinical note snippet that would answer this question 
about a patient's condition. Include diagnosis date, current status, and treatment if relevant.

Question: {query}

Clinical note:""",

    "medication": """Generate a realistic medication record entry that would answer this question.
Include drug name, dosage, frequency, and start date.

Question: {query}

Medication record:""",

    "lab_result": """Generate a realistic lab result entry that would answer this question.
Include test name, value, units, reference range, and date.

Question: {query}

Lab result:""",

    "vitals": """Generate a realistic vital signs entry that would answer this question.
Include measurement, value, units, and date.

Question: {query}

Vital signs:""",

    "default": """You are a medical records system. Generate a realistic medical record entry 
that would answer this question. Be specific and concise.

Question: {query}

Medical record entry:"""
}


@dataclass
class HyDEResult:
    """Result from HyDE retrieval."""
    memories: List[Dict]
    latency_ms: float
    hypothetical_doc: str
    query_type: str


class HyDERetrieval:
    """
    Hypothetical Document Embeddings for medical memory retrieval.
    
    Instead of embedding the question directly, we:
    1. Use LLM to generate a hypothetical answer
    2. Embed the hypothetical answer
    3. Search using that embedding
    
    This bridges the query-document gap in embedding space.
    """
    
    name = "hyde"
    description = "Hypothetical Document Embeddings - generates fake answer for better retrieval"
    
    def __init__(self, config: Dict = None, num_hypotheticals: int = 1):
        """
        Initialize HyDE retrieval.
        
        Args:
            config: Mem0 configuration
            num_hypotheticals: Number of hypothetical docs to generate and average
        """
        self.config = config or self._default_config()
        self.memory = Memory.from_config(self.config)
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.num_hypotheticals = num_hypotheticals
    
    def _default_config(self) -> Dict:
        return {
            "vector_store": {
                "provider": "pinecone",
                "config": {
                    "api_key": os.getenv("PINECONE_API_KEY"),
                    "collection_name": os.getenv("PINECONE_INDEX_NAME", "medmem0"),
                    "embedding_model_dims": 1536,
                    "serverless_config": {
                        "cloud": "aws",
                        "region": "us-east-1",
                    }
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                }
            }
        }
    
    def _detect_query_type(self, query: str) -> str:
        """Detect query type for appropriate prompt template."""
        query_lower = query.lower()
        
        # Medication queries
        if any(kw in query_lower for kw in ["medication", "drug", "prescription", "taking", "prescribed"]):
            return "medication"
        
        # Lab queries
        if any(kw in query_lower for kw in ["lab", "a1c", "glucose", "cholesterol", "creatinine", "test result"]):
            return "lab_result"
        
        # Vital queries
        if any(kw in query_lower for kw in ["blood pressure", "heart rate", "weight", "bmi", "vital", "temperature"]):
            return "vitals"
        
        # Condition queries
        if any(kw in query_lower for kw in ["diabetes", "hypertension", "condition", "diagnosis", "disease"]):
            return "condition"
        
        # History queries
        if any(kw in query_lower for kw in ["history", "when", "first", "diagnosed"]):
            return "medical_history"
        
        return "default"
    
    def _generate_hypothetical(self, query: str, query_type: str) -> str:
        """Generate a hypothetical document using LLM."""
        prompt_template = HYDE_PROMPTS.get(query_type, HYDE_PROMPTS["default"])
        prompt = prompt_template.format(query=query)
        
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You generate realistic medical record entries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Some variation for diversity
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
    
    def _generate_multiple_hypotheticals(self, query: str, query_type: str, n: int) -> List[str]:
        """Generate multiple hypothetical documents for averaging."""
        hypotheticals = []
        
        for _ in range(n):
            hypo = self._generate_hypothetical(query, query_type)
            hypotheticals.append(hypo)
        
        return hypotheticals
    
    def search(
        self, 
        query: str, 
        patient_id: str, 
        k: int = 5
    ) -> Tuple[List[Dict], float]:
        """
        Search using HyDE.
        
        Args:
            query: User's question
            patient_id: Patient ID
            k: Number of results
        
        Returns:
            (memories, latency_ms)
        """
        start = time.perf_counter()
        
        # 1. Detect query type
        query_type = self._detect_query_type(query)
        
        # 2. Generate hypothetical document(s)
        if self.num_hypotheticals == 1:
            hypothetical = self._generate_hypothetical(query, query_type)
            search_text = hypothetical
        else:
            hypotheticals = self._generate_multiple_hypotheticals(query, query_type, self.num_hypotheticals)
            # Combine for search (Mem0 will embed this)
            search_text = " ".join(hypotheticals)
        
        # 3. Search using hypothetical document
        results = self.memory.search(
            query=search_text,
            user_id=patient_id,
            limit=k
        )
        
        # Handle Mem0 response format {'results': [...]}
        if isinstance(results, dict):
            results = results.get('results', [])
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        # Normalize results
        memories = []
        for r in (results or []):
            if isinstance(r, dict):
                memories.append({
                    "id": r.get("id", ""),
                    "content": r.get("memory", r.get("content", "")),
                    "memory": r.get("memory", r.get("content", "")),
                    "metadata": r.get("metadata", {}),
                    "score": r.get("score")
                })
        
        return memories, latency_ms
    
    def search_with_details(
        self, 
        query: str, 
        patient_id: str, 
        k: int = 5
    ) -> HyDEResult:
        """Search with full details including generated hypothetical."""
        start = time.perf_counter()
        
        query_type = self._detect_query_type(query)
        hypothetical = self._generate_hypothetical(query, query_type)
        
        results = self.memory.search(
            query=hypothetical,
            user_id=patient_id,
            limit=k
        )
        
        # Handle Mem0 response format {'results': [...]}
        if isinstance(results, dict):
            results = results.get('results', [])
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        memories = []
        for r in (results or []):
            if isinstance(r, dict):
                memories.append({
                    "id": r.get("id", ""),
                    "content": r.get("memory", r.get("content", "")),
                    "metadata": r.get("metadata", {}),
                    "score": r.get("score")
                })
        
        return HyDEResult(
            memories=memories,
            latency_ms=latency_ms,
            hypothetical_doc=hypothetical,
            query_type=query_type
        )
    
    def compare_with_vanilla(
        self, 
        query: str, 
        patient_id: str, 
        k: int = 5
    ) -> Dict:
        """Compare HyDE vs vanilla retrieval."""
        
        # Vanilla search
        vanilla_start = time.perf_counter()
        vanilla_results = self.memory.search(query=query, user_id=patient_id, limit=k)
        vanilla_latency = (time.perf_counter() - vanilla_start) * 1000
        
        # Handle Mem0 response format
        if isinstance(vanilla_results, dict):
            vanilla_results = vanilla_results.get('results', [])
        
        # HyDE search
        hyde_result = self.search_with_details(query, patient_id, k)
        
        return {
            "query": query,
            "vanilla": {
                "results": [r.get("memory", r.get("content", ""))[:100] for r in (vanilla_results or []) if isinstance(r, dict)],
                "latency_ms": vanilla_latency
            },
            "hyde": {
                "hypothetical": hyde_result.hypothetical_doc,
                "query_type": hyde_result.query_type,
                "results": [m["content"][:100] for m in hyde_result.memories],
                "latency_ms": hyde_result.latency_ms
            }
        }
    
    def explain(self, query: str) -> Dict:
        """Explain what HyDE would do for a query (without searching)."""
        query_type = self._detect_query_type(query)
        hypothetical = self._generate_hypothetical(query, query_type)
        
        return {
            "original_query": query,
            "detected_type": query_type,
            "prompt_used": HYDE_PROMPTS[query_type][:100] + "...",
            "generated_hypothetical": hypothetical,
            "explanation": (
                f"Instead of searching for '{query}', HyDE will search using "
                f"the hypothetical document above. This hypothetical is closer "
                f"in embedding space to real medical records."
            )
        }


# Convenience function
def create_hyde_retrieval(config: Dict = None, num_hypotheticals: int = 1) -> HyDERetrieval:
    """Create a HyDE retrieval instance."""
    return HyDERetrieval(config, num_hypotheticals)


# For compatibility with experiments framework
class HyDE(HyDERetrieval):
    """Alias for use in experiments."""
    name = "hyde"
    description = "Hypothetical Document Embeddings"
