from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()

def create_vector_store(documents, index_path):
    vector_store = FAISS.from_documents(
        documents, embeddings
    )
    vector_store.save_local(index_path)


def load_vector_store(index_path):
    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
    return None