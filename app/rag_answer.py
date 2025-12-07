import json
from rag_engine import RAGEngine
from config import FACT_CHECK_PROMPT
from google.genai  import GenerativeModel

class RAGAnswer:
    def __init__(self):
        self.engine = RAGEngine()
        self.model = GenerativeModel("gemini-2.0-flash")

    def build_context(self, retrieved):
        return "\n\n".join([c["text"] for c in retrieved])

    def fact_check(self, question: str):
        retrieved = self.engine.retrieve(question, k=4)
        context = self.build_context(retrieved)

        prompt = FACT_CHECK_PROMPT.format(
            question=question,
            context=context
        )

        response = self.model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        return json.loads(response.text)
