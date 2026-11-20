import hashlib
import json
from pathlib import Path
from typing import List
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

CHROMA_DIR = "./chroma_db"
CHUNKS_CACHE = "./data/processed_chunks.jsonl"
COLLECTION_NAME = "domain_docs"

def load_documents(raw_dir: str = "./data/raw_docs") -> List[Document]:
    docs: List[Document] = []
    pdf_loader = DirectoryLoader(raw_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    md_loader = DirectoryLoader(raw_dir, glob="**/*.md", loader_cls=TextLoader)
    docs.extend(pdf_loader.load())
    docs.extend(md_loader.load())
    return docs

def chunk_documents(docs: List[Document]) -> List[Document]:
    # ~650 tokens (approx 2600 chars) with 100-token (~400 char) overlap
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=650,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for idx, chunk in enumerate(chunks):
        source = Path(chunk.metadata.get("source", "unknown")).name
        page = chunk.metadata.get("page", 0)
        digest = hashlib.sha256(f"{source}:{page}:{idx}".encode()).hexdigest()[:8]
        chunk.metadata["chunk_id"] = f"{source}:p{page}:c{idx}_{digest}"
        chunk.metadata["source_file"] = source
        chunk.metadata["paragraph_index"] = idx

    return chunks

def build_index(raw_dir: str = "./data/raw_docs") -> None:
    raw_docs = load_documents(raw_dir)
    chunks = chunk_documents(raw_docs)

    # Save chunks locally so BM25 can load the exact same corpus in Phase 2
    Path(CHUNKS_CACHE).parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_CACHE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"page_content": c.page_content, "metadata": c.metadata}) + "\n")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Indexed {len(chunks)} chunks into ChromaDB at {CHROMA_DIR}")

if __name__ == "__main__":
    build_index()
