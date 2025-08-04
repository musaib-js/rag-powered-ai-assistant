import gradio as gr
import uuid
from doc_loaders import load_documents
from datetime import datetime
import os
import pymongo
from dotenv import load_dotenv
from langchain_openai import OpenAI
from pinecone_vec import create_vector_store, query_vector_store

load_dotenv()

# Setup MongoDB
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client["document_store"]
documents_collection = db["documents"]
bot_collection = db["bots"]
messages_collection = db["messages"]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

llm = OpenAI()



def create_bot(bot_name, files, bot_metadata):
    if not files:
        return "Please upload at least one file."

    document_id = str(uuid.uuid4())

    bot_id = str(uuid.uuid4())
    bot_data = {
        "bot_id": bot_id,
        "bot_name": bot_name,
        "bot_metadata": bot_metadata,
        "file_id": document_id,
    }
    bot_collection.insert_one(bot_data)

    for file in files:
        if isinstance(file, str):
            file_path = file
            file_type = os.path.splitext(file_path)[-1].lower().replace(".", "")
            document_id = str(uuid.uuid4())
        else:
            file_type = os.path.splitext(file.name)[-1].lower().replace(".", "")
            document_id = str(uuid.uuid4())
            file_path = os.path.join(UPLOAD_FOLDER, f"{document_id}.{file_type}")
            with open(file_path, "wb") as f_out:
                f_out.write(file.read())

        documents = load_documents(file_path, file_type)
        create_vector_store(documents, document_id, bot_id)

        if not isinstance(file, str):
            os.remove(file_path)

        document_data = {
            "file_name": files[0].name,
            "file_type": file_type,
            "document_id": document_id,
            "file_id": document_id,
            "bot_id": bot_id,
        }
        documents_collection.insert_one(document_data)

    messages_collection.insert_one({"bot_id": bot_id, "messages": []})

    return f"Bot: '{bot_name}' created with ID: {bot_id}"


def get_bots():
    files = list(bot_collection.find({}, {"_id": 0, "bot_name": 1, "bot_id": 1}))
    bots_dict = {file["bot_name"]: file["bot_id"] for file in files}
    return bots_dict


def load_bot_chat(bot_id):
    document = messages_collection.find_one({"bot": bot_id})
    if not document:
        return []
    messages = document.get("messages", [])
    return [
        (m["message"], None if m["from"] == "user" else m["message"]) for m in messages
    ]


def handle_query_stream(query, bot_id, history):
    bot = bot_collection.find_one({"bot_id": bot_id})
    if not bot:
        yield history + [(query, "Bot not found.")]
        return

    matches = query_vector_store(bot_id, query)

    if not matches:
        yield history + [(query, "No relevant documents found.")]
        return

    context = "\n\n".join([match["text"] for match in matches])
    prompt = f"""You are a helpful assistant of {bot["bot_name"]} - {bot["bot_metadata"]}. Answer the question based on the context below:

Context:
{context}

Question: {query}

The answer should be polite and concise, always asking if the user needs further assistance. If the context is not sufficient, politely inform the user that that the requested information is not available and ask if they need help with something else and give them an idea of what they can ask about."""

    # Stream the response
    response_chunks = []
    streamed_text = ""
    start_time = datetime.now()
    
    for chunk in llm.stream(prompt):
        streamed_text += str(chunk) 
        response_chunks.append(str(chunk))
        yield history + [(query, streamed_text)]

    end_time = datetime.now()

    message_1 = {
        "from": "user",
        "message": query,
        "timestamp": start_time.isoformat(),
    }
    message_2 = {
        "from": "assistant",
        "message": streamed_text,
        "timestamp": end_time.isoformat(),
    }

    messages_collection.update_one(
        {"bot_id": bot_id},
        {"$push": {"messages": {"$each": [message_1, message_2]}}},
    )


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # 🤖 Chatbot Q&A System (Pinecone + OpenAI + Gradio)
    Create and chat with document-based AI assistants.
    """
    )

    with gr.Tab("🆕 Create Chatbot"):
        bot_name = gr.Textbox(label="Bot Name")
        bot_metadata = gr.Textbox(label="Bot Metadata (JSON or description)")
        file_uploader = gr.File(
            label="Upload Documents",
            file_types=[".pdf", ".txt", ".csv"],
            file_count="multiple",
        )
        create_btn = gr.Button("Create Bot")
        create_status = gr.Textbox(label="Status")

        create_btn.click(
            create_bot,
            inputs=[bot_name, file_uploader, bot_metadata],
            outputs=create_status,
        )

    with gr.Tab("💬 Chat with Bot"):

        bot_selector = gr.Dropdown(label="Select Bot", choices=[], interactive=True)
        refresh_btn = gr.Button("🔄 Refresh Bots")

        chatbox = gr.Chatbot(label="Conversation")
        user_input = gr.Textbox(label="Your question")
        ask_btn = gr.Button("Ask")

        bot_id_state = gr.State()
        bots_dict_state = gr.State({})

        def refresh_bot_list():
            bots_list = get_bots()
            return gr.update(choices=list(bots_list.keys())), bots_list

        refresh_btn.click(refresh_bot_list, outputs=[bot_selector, bots_dict_state])

        def select_bot(bot_name, bots_dict):
            bot_id = bots_dict.get(bot_name)
            history = load_bot_chat(bot_id)
            return bot_id, history

        bot_selector.change(
            select_bot,
            inputs=[bot_selector, bots_dict_state],
            outputs=[bot_id_state, chatbox],
        )

        ask_btn.click(
            handle_query_stream,
            inputs=[user_input, bot_id_state, chatbox],
            outputs=chatbox,
        )

# ---- Launch ----
demo.launch(share=True)
