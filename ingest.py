import os
import re

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

SOURCE_URL = "https://statutes.capitol.texas.gov/Docs/PR/htm/PR.92.htm"


def load_statute_text(url: str) -> str:
    """Download the statute page and return clean text."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text("\n")
    text = text.replace("\xa0", " ")
    return re.sub(r"\n\s*\n+", "\n", text)


def split_into_sections(text: str) -> list[Document]:
    """Split the chapter into one Document per statute section, e.g. 'Sec. 92.102'."""
    parts = re.split(r"(?=^Sec\.\s+92\.\d+\.\s)", text, flags=re.MULTILINE)
    docs = []
    for part in parts:
        match = re.match(r"Sec\.\s+(92\.\d+)\.\s+([^.]+)\.", part)
        if not match:
            continue
        section_number, title = match.group(1), match.group(2).strip().title()
        docs.append(
            Document(
                page_content=part.strip(),
                metadata={
                    "section": section_number,
                    "title": title,
                    "source": f"{SOURCE_URL}#{section_number}",
                },
            )
        )
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Long sections get split further; metadata is kept on every chunk."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    return splitter.split_documents(docs)


if __name__ == "__main__":
    print("Downloading Texas Property Code Chapter 92...")
    raw_text = load_statute_text(SOURCE_URL)

    sections = split_into_sections(raw_text)
    print(f"Found {len(sections)} sections")

    chunks = chunk_documents(sections)
    print(f"Created {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    PineconeVectorStore.from_documents(
        chunks, embeddings, index_name=os.environ["INDEX_NAME"]
    )
    print("Done. Chunks stored in Pinecone.")
