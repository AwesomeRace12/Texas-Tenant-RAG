import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_llama4 = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

prompt_template = ChatPromptTemplate.from_template(
    """You are an assistant that explains Texas residential tenant law.
Use ONLY the statute excerpts below to answer. Cite the section number
in brackets, like [Sec. 92.103], after each claim. If the excerpts do not
contain the answer, say you could not find it in Chapter 92.

Statute excerpts:
{context}

Question: {question}

Answer in plain English:"""
)

DISCLAIMER = "\n\nNote: This is general information, not legal advice."


def format_docs(docs):
    """Label each chunk with its section so the model can cite it."""
    return "\n\n".join(
        f"[Sec. {doc.metadata.get('section', '?')} - {doc.metadata.get('title', '')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )


# ============================================================================
# VERSION 1: Without LCEL (step by step)
# ============================================================================


def answer_without_lcel(question: str, model=llm) -> str:
    """Retrieve, format, prompt, and call the model manually."""
    docs = retriever.invoke(question)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=question)
    response = model.invoke(messages)
    return response.content + DISCLAIMER


# ============================================================================
# VERSION 2: With LCEL (chained)
# ============================================================================


def build_chain(model=llm):
    """Return a chain invoked with {"question": "..."}."""
    return (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | model
        | StrOutputParser()
    )


def get_sources(question: str) -> list[str]:
    """List the statute sections used for an answer (for a 'Sources' display)."""
    docs = retriever.invoke(question)
    seen = []
    for doc in docs:
        label = f"Sec. {doc.metadata.get('section')} - {doc.metadata.get('title')}"
        if label not in seen:
            seen.append(label)
    return seen


# ============================================================================
# Model comparison on a small test set
# ============================================================================

TEST_QUESTIONS = [
    "How many days does my landlord have to return my security deposit?",
    "Can my landlord lock me out if I am late on rent?",
    "When can a landlord charge a late fee?",
    "Can I break my lease if I am a victim of family violence?",
    "Does my landlord have to repair my broken air conditioner?",
]


def compare_models():
    chain_llama33 = build_chain(llm)
    chain_llama4 = build_chain(llm_llama4)
    for q in TEST_QUESTIONS:
        print("=" * 70)
        print("Q:", q)
        print("\n--- Llama 3.3 70B ---")
        print(chain_llama33.invoke({"question": q}))
        print("\n--- Llama 4 Scout ---")
        print(chain_llama4.invoke({"question": q}))


if __name__ == "__main__":
    question = "How many days does my landlord have to return my security deposit?"

    print("\n" + "=" * 70)
    print("VERSION 0: Plain LLM (no retrieval)")
    print("=" * 70)
    print(llm.invoke([HumanMessage(content=question)]).content)

    print("\n" + "=" * 70)
    print("VERSION 1: RAG without LCEL")
    print("=" * 70)
    print(answer_without_lcel(question))

    print("\n" + "=" * 70)
    print("VERSION 2: RAG with LCEL")
    print("=" * 70)
    print(build_chain().invoke({"question": question}) + DISCLAIMER)

    print("\nSources used:")
    for source in get_sources(question):
        print(" -", source)

    # Uncomment to compare both Llama models side by side:
    # compare_models()
