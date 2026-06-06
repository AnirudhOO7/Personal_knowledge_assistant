import json 
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from app.core.config import settings
from app.core.logging import get_logger
from app.llm.prompt import RAG_PROMPT, REFLECTION_PROMPT
from app.retrieval.retriever import format_context
from pathlib import Path

logger = get_logger(__name__)

_llm = None

def get_llm()->ChatAnthropic:
    """Return a singleton llm instance"""
    global _llm

    if _llm is None:
        logger.info(f"Loading LLM: {settings.llm_model}")
        _llm = ChatAnthropic(
            model= settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.anthropic_api_key
        )

        logger.info("LLM loaded")

    return _llm



def reflect(question:str, docs:list[Document])->dict:
    """
    Evaluate if retrieved context is sufficient to answer the question.
    Returns dict with is_sufficient, reason, and missing fields.
    """

    llm=get_llm()
    context=format_context(docs)

    chain = REFLECTION_PROMPT | llm
    response = chain.invoke({
    "question": question,
    "context": context
    })
    

    try:
        result = json.loads(response.content)
        logger.info(f"Reflection result: sufficient={result.get('is_sufficient')} | reason={result.get('reason')}")
        return result
    except json.JSONDecodeError:
        logger.warning("Reflection response was not valid JSON, defaulting to sufficient")
        return {"is_sufficient": True, "reason": "parse error", "missing": None}


def generate(question:str , docs: list[Document])->str:
    """Generate a grounded answer from the retrieved docs."""
    llm=get_llm()
    content=format_context(docs)

    logger.info(f"Generating answer for: '{question}'")

    chain = RAG_PROMPT | llm
    response = chain.invoke({
    "question": question,
    "context": content
    })

    return response.content