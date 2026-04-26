import pytest

from src.chain.prompts import build_prompt


def test_prompt_template_renders():
    prompt = build_prompt()
    rendered = prompt.format_messages(context="some context", question="What is X?")
    assert any("some context" in m.content for m in rendered)
    assert any("What is X?" in m.content for m in rendered)


@pytest.mark.integration
def test_rag_chain_end_to_end():
    """Requires populated vector store + GROQ_API_KEY."""
    from src.chain.rag_chain import build_rag_chain

    chain = build_rag_chain(ticker="AAPL", filing_type="10-K")
    out = chain.invoke("What does the company say about risks?")
    assert "answer" in out
    assert isinstance(out["sources"], list)
