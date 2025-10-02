from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import os, time, json, urllib.parse, urllib.request

# LLM integration
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

@dataclass
class ToolResult:
    name: str
    args: Dict[str, Any]
    stdout: str = ""     # main payload (JSON string)
    stderr: str = ""
    returncode: int = 0
    latency_ms: Optional[int] = None

def _http_get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    qs = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"{url}?{qs}", timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))

class BM25SearchTool:
    """
    Optional HTTP client to a BM25 retrieval service.
    Set env BM25_API_URL to something like: http://localhost:8001/search
    Expected JSON: {"docs":[{"title": "...", "text":"..."}, ...]}
    """
    def __init__(self):
        self.name = "bm25_search"
        self.url = os.environ.get("BM25_API_URL")  # None means stub

    def __call__(self, question: str, k: int = 5) -> ToolResult:
        t0 = time.time()
        try:
            if not self.url:
                # Stub fallback
                docs = [{"title":"Stub BM25", "text":f"No BM25 service. Fallback for: {question}"}]
                return ToolResult(self.name, {"q": question, "k": k},
                                  stdout=json.dumps({"docs": docs}),
                                  latency_ms=int((time.time()-t0)*1000))
            data = _http_get_json(self.url, {"q": question, "k": k})
            return ToolResult(self.name, {"q": question, "k": k},
                              stdout=json.dumps(data),
                              latency_ms=int((time.time()-t0)*1000))
        except Exception as e:
            return ToolResult(self.name, {"q": question, "k": k}, stderr=str(e), returncode=2)

class DenseSearchTool:
    """
    Optional HTTP client to a dense retrieval API (e.g., Contriever/BGE).
    Set env DENSE_API_URL, e.g., http://localhost:8002/search
    Expected JSON: {"docs":[{"title": "...", "text":"..."}, ...]}
    """
    def __init__(self):
        self.name = "dense_search"
        self.url = os.environ.get("DENSE_API_URL")

    def __call__(self, question: str, k: int = 5) -> ToolResult:
        t0 = time.time()
        try:
            if not self.url:
                docs = [{"title":"Stub Dense", "text":f"No dense service. Fallback for: {question}"}]
                return ToolResult(self.name, {"q": question, "k": k},
                                  stdout=json.dumps({"docs": docs}),
                                  latency_ms=int((time.time()-t0)*1000))
            data = _http_get_json(self.url, {"q": question, "k": k})
            return ToolResult(self.name, {"q": question, "k": k},
                              stdout=json.dumps(data),
                              latency_ms=int((time.time()-t0)*1000))
        except Exception as e:
            return ToolResult(self.name, {"q": question, "k": k}, stderr=str(e), returncode=2)

class HybridMergeTool:
    """Merge BM25 + Dense results with simple title+text dedupe and top-K cut."""
    def __init__(self): self.name = "hybrid_merge"

    def __call__(self, bm25_json: str, dense_json: str, k: int = 10) -> ToolResult:
        import hashlib
        t0 = time.time()
        try:
            b = json.loads(bm25_json).get("docs", [])
            d = json.loads(dense_json).get("docs", [])
            seen, merged = set(), []
            for doc in (b + d):
                key = hashlib.md5((doc.get("title","") + "|" + doc.get("text","")).encode("utf-8")).hexdigest()
                if key not in seen:
                    merged.append(doc)
                    seen.add(key)
                if len(merged) >= k:
                    break
            return ToolResult(self.name, {"k": k},
                              stdout=json.dumps({"docs": merged}),
                              latency_ms=int((time.time()-t0)*1000))
        except Exception as e:
            return ToolResult(self.name, {"k": k}, stderr=str(e), returncode=2)

class HeuristicReader:
    """
    Very light reader: pick the first sentence containing a named entity-ish token
    or a number; otherwise take the first ~12 words from the top doc. Pure stub,
    but deterministic and useful for trajectory shape.
    """
    def __init__(self): self.name = "heuristic_reader"

    def __call__(self, question: str, merged_json: str) -> ToolResult:
        import re
        t0 = time.time()
        try:
            docs = json.loads(merged_json).get("docs", [])
            text = " ".join([d.get("text","") for d in docs[:1]])  # top doc only
            # crude sentence split
            sents = re.split(r"(?<=[.!?])\s+", text)
            pick = None
            for s in sents:
                if re.search(r"[A-Z][a-z]{2,}", s) or re.search(r"\d", s):
                    pick = s.strip()
                    break
            if not pick:
                pick = " ".join(text.split()[:12]).strip() or "N/A"
            return ToolResult(self.name, {"q": question}, stdout=json.dumps({"answer": pick}),
                              latency_ms=int((time.time()-t0)*1000))
        except Exception as e:
            return ToolResult(self.name, {"q": question}, stderr=str(e), returncode=2)

class LLMReader:
    """
    LLM-powered reader that uses OpenAI GPT models for intelligent multi-hop reasoning.
    Falls back to HeuristicReader if OpenAI is not available or API key is missing.
    """
    def __init__(self, model: str = "gpt-4o-mini"):
        self.name = "llm_reader"
        self.model = model
        self.client = None
        self.fallback_reader = HeuristicReader()
        
    def _get_client(self):
        if self.client is None:
            if OpenAI is None:
                raise RuntimeError("OpenAI client not available. Install openai>=1.40")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            self.client = OpenAI(api_key=api_key)
        return self.client

    def __call__(self, question: str, merged_json: str) -> ToolResult:
        """Use LLM to reason about the question and context to find the answer"""
        t0 = time.time()
        
        try:
            # Parse the merged context
            docs = json.loads(merged_json).get("docs", [])
            if not docs:
                return self.fallback_reader(question, merged_json)
            
            # Build context from retrieved documents
            context_parts = []
            for i, doc in enumerate(docs[:5]):  # Use top 5 docs
                title = doc.get("title", f"Document {i+1}")
                text = doc.get("text", "")
                context_parts.append(f"**{title}**: {text}")
            
            context = "\n\n".join(context_parts)
            
            # Get LLM client
            client = self._get_client()
            
            # Create a reasoning prompt
            prompt = f"""You are a multi-hop question answering expert. Given a question and retrieved context, provide a step-by-step reasoning process to find the answer.

Question: {question}

Context:
{context}

Instructions:
1. Analyze the question to understand what information is needed
2. Examine the context to find relevant information
3. Connect information from multiple sources if needed (multi-hop reasoning)
4. Provide your reasoning step by step
5. Give a clear, concise final answer

Format your response as:
REASONING: [Your step-by-step analysis]
ANSWER: [The final answer]"""

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            result = response.choices[0].message.content.strip()
            
            # Extract answer from response
            if "ANSWER:" in result:
                answer = result.split("ANSWER:")[-1].strip()
                reasoning = result.split("ANSWER:")[0].replace("REASONING:", "").strip()
            else:
                answer = result
                reasoning = "No explicit reasoning provided"
                
            return ToolResult(
                self.name,
                {"question": question, "model": self.model},
                stdout=json.dumps({"answer": answer, "reasoning": reasoning, "full_response": result}),
                latency_ms=int((time.time() - t0) * 1000)
            )
            
        except Exception as e:
            # Fallback to heuristic reader
            print(f"⚠️ LLM Reader failed: {e}. Falling back to heuristic reader.")
            return self.fallback_reader(question, merged_json)
