# Texas Tenant Rights Assistant (RAG)

A retrieval-augmented generation (RAG) app that answers questions about Texas
residential tenant law, with answers grounded in and cited to specific sections
of [Texas Property Code Chapter 92](https://statutes.capitol.texas.gov/Docs/PR/htm/PR.92.htm).

## Stack

- **LangChain** (LCEL) for the retrieval chain
- **Pinecone** as the vector store
- **OpenAI** `text-embedding-3-small` for embeddings
- **Groq** for LLM inference (Llama 3.3 70B and Llama 4 Scout)

## How it works

1. `ingest.py` downloads Chapter 92, splits it by statute section, chunks long
   sections, and stores the chunks (with section metadata) in Pinecone.
2. `main.py` retrieves the top matching chunks for a question, passes them to
   the LLM with a prompt that requires section citations, and returns the answer.
   It includes a no-RAG baseline, a step-by-step version, an LCEL version, and a
   side-by-side comparison of the two Llama models.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your keys
```

Create a Pinecone index named as in `INDEX_NAME` with dimension **1536** and the
**cosine** metric.

```bash
python ingest.py   # run once
python main.py
```

## Disclaimer

This project provides general information only and is not legal advice.

## Acknowledgments

Project structure inspired by Eden Marco's LangChain course.
