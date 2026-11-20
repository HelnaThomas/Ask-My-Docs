from typing import List, Tuple
import yaml
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

class ClaimCitation(BaseModel):
    claim: str = Field(..., description="Single factual statement answering part of the question.")
    chunk_id: str = Field(..., description="Exact chunk_id from the retrieved evidence.")
    verbatim_quote: str = Field(..., description="Exact substring copied from the cited chunk.")

class CitationEnforcedResponse(BaseModel):
    can_answer: bool = Field(..., description="True ONLY if the evidence directly supports the answer.")
    claims: List[ClaimCitation] = Field(default_factory=list)
    answer: str = Field(..., description="Final response with inline [chunk_id] citations, or explicit refusal.")

def load_prompt_config(path: str = "config/prompts.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class GroundedGenerator:
    def __init__(self, config_path: str = "config/prompts.yaml"):
        self.config = load_prompt_config(config_path)
        self.prompt_version = self.config["version"]
        self.llm = ChatOpenAI(
            model=self.config["model"],
            temperature=self.config["temperature"],
        ).with_structured_output(CitationEnforcedResponse)

    def generate(self, question: str, retrieved: List[Tuple[Document, float]]) -> CitationEnforcedResponse:
        if not retrieved:
            return CitationEnforcedResponse(
                can_answer=False,
                claims=[],
                answer="I cannot answer this question based on the provided documents (no relevant evidence met the confidence threshold).",
            )

        valid_chunks = {doc.metadata["chunk_id"]: doc.page_content for doc, _ in retrieved}
        formatted_context = "\n\n".join(
            f"[{doc.metadata['chunk_id']}] (Source: {doc.metadata['source_file']}, Score: {score:.2f})\n{doc.page_content}"
            for doc, score in retrieved
        )

        prompt_cfg = self.config["prompts"]["grounded_qa"]
        messages = [
            ("system", prompt_cfg["system"]),
            ("user", prompt_cfg["user_template"].format(context=formatted_context, question=question)),
        ]

        response: CitationEnforcedResponse = self.llm.invoke(messages)

        # Hard guardrail: reject hallucinated chunk IDs or fabricated quotes
        if response.can_answer:
            if not response.claims:
                return CitationEnforcedResponse(
                    can_answer=False,
                    claims=[],
                    answer="I cannot answer this question based on the provided documents (no verifiable citations produced).",
                )
            for c in response.claims:
                if c.chunk_id not in valid_chunks:
                    return CitationEnforcedResponse(
                        can_answer=False,
                        claims=[],
                        answer="I cannot answer this question based on the provided documents (citation verification failed).",
                    )
                if c.verbatim_quote.strip().lower() not in valid_chunks[c.chunk_id].lower():
                    return CitationEnforcedResponse(
                        can_answer=False,
                        claims=[],
                        answer="I cannot answer this question based on the provided documents (quote verification failed).",
                    )

        return response
