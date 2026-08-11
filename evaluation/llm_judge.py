import os
from pathlib import Path

from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / "env" / ".env")

JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "openai/gpt-4o-mini")


class OpenRouterJudge(DeepEvalBaseLLM):
    """DeepEval-compatible LLM judge backed by OpenRouter, reusing OPENROUTER_API_KEY
    so evaluation doesn't require a separate OpenAI key."""

    def __init__(self, model_name: str = JUDGE_MODEL):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY belum di-set di env/.env")
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return res.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.model_name


def build_ragas_llm():
    """Wraps the same OpenRouter judge model for RAGAS via its LangChain ChatOpenAI wrapper."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY belum di-set di env/.env")
    return ChatOpenAI(
        model=JUDGE_MODEL,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
