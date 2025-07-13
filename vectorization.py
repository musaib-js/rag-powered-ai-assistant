import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from tqdm import tqdm
import logging
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import uuid

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT,
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

index = pc.Index(PINECONE_INDEX_NAME)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)


def create_vector_store(documents, document_id, bot_id):
    split_docs = text_splitter.split_documents(documents)

    for i, doc in enumerate(split_docs):

        metadata = {
            "bot_id": bot_id,
            "chunk_id": f"{bot_id}_{document_id}_chunk_{i}",
            "text": doc.page_content,
        }

        vector = embeddings.embed_query(doc.page_content)

        index.upsert(
            [{"id": str(uuid.uuid4()), "values": vector, "metadata": metadata}]
        )

def query_vector_store(bot_id: str, query: str, top_k=5):
    
    query_vector = embeddings.embed_query(query)

    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter={
            "bot_id": bot_id
        }
    )
    
    print(response)
    results = [
        {
            "score": match["score"],
            "text": match["metadata"].get("text"),
            "chunk_id": match["metadata"].get("chunk_id"),
            "bot_id": match["metadata"].get("bot_id")
        }
        for match in response["matches"]
    ]
    
    logging.info(f"Found {len(results)} results for topic '{query}'")
    
    return results

