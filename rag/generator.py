# rag/generator.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env variables
load_dotenv()

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("No GEMINI_API_KEY found in environment variables.")

class GeminiGenerator:
    def __init__(self, index, model="gemini-1.5-flash"):
        self.index = index
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(model)

    def _build_prompt(self, question, contexts):
        ctx_text = "\n\n".join([f"[SOURCE {i+1}]\n{c}" for i, c in enumerate(contexts)])
        return f"""
You are a precise and step-by-step tutor for math and science.
Use ONLY the provided context to answer the question.
ALWAYS show reasoning step-by-step and then give the final answer clearly.
If the limit diverges, say "Diverges to +∞" or "Does not exist" clearly.

Context:
{ctx_text}

Question:
{question}

Answer in LaTeX-style formatting when writing math.
"""

    def generate(self, question, top_k=3):
        contexts = [m["text"] for (_score, m) in self.index.query(question, top_k=top_k)]
        prompt = self._build_prompt(question, contexts)
        response = self.model.generate_content(prompt)
        return response.text
