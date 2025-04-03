from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAI
import os
import uuid
from dotenv import load_dotenv
from flask_restx import Api, Resource
from request_validators import register_models
import pymongo
from googledrive.google_drive_handler import upload_to_google_drive
from doc_loaders import load_documents
from vectorisation import create_vector_store, load_vector_store
from datetime import datetime
import time

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)
api = Api(
    app,
    version="1.0",
    title="Document Q&A API",
    description="API for Document Question Answering",
)
CORS(app)


MONGO_URI = os.getenv("MONGO_URI")
mongo_client = pymongo.MongoClient(MONGO_URI)
db = mongo_client["document_store"]
documents_collection = db["documents"]
logs_collection = db["logs"]

UPLOAD_FOLDER = "uploads"
INDEX_FOLDER = "indexes"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

upload_parser, upload_model, query_model = register_models(api)


def get_qa_chain(vector_store):
    retriever = vector_store.as_retriever()
    return RetrievalQA.from_chain_type(llm=OpenAI(), retriever=retriever)


@api.route("/upload")
class Upload(Resource):
    def post(self):
        file = request.files["file"]
        file_type = file.filename.split(".")[-1].lower()

        if file_type not in ["pdf", "csv", "txt"]:
            return (
                jsonify({"error": "Invalid file type. Only PDF, CSV, and TXT allowed"}),
                400,
            )

        document_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f"{document_id}.{file_type}")
        file.save(file_path)

        # Upload to Google Drive
        file_id = upload_to_google_drive(file_path, file.filename)

        # Store metadata in MongoDB
        document_data = {
            "file_name": file.filename,
            "file_id": file_id,
            "file_type": file_type,
            "document_id": document_id,
            "messages": [],
        }
        documents_collection.insert_one(document_data)

        documents = load_documents(file_path, file_type)
        index_path = os.path.join(INDEX_FOLDER, document_id)
        create_vector_store(documents, index_path)

        os.remove(file_path)

        return jsonify(
            {
                "message": "File uploaded and indexed successfully",
                "document_id": document_id,
            }
        )


@api.route("/query")
class Query(Resource):
    @api.expect(query_model)
    def post(self):
        file_id = request.json["file_id"]
        document_data = documents_collection.find_one({"file_id": file_id})
        if not document_data:
            return {"error": "Document not found"}, 404

        index_path = os.path.join(INDEX_FOLDER, document_data["document_id"])
        vector_store = load_vector_store(index_path)
        if vector_store is None:
            return {"error": "No index found for this document"}, 400

        data = request.json
        query_text = data["query"]

        start_time = time.time()  # Start time for latency tracking

        qa_chain = get_qa_chain(vector_store)
        response = qa_chain.invoke(query_text)

        response_time = time.time() - start_time  # Calculate response time

        # Create tracking logs
        log_entry = {
            "file_id": file_id,
            "query": query_text,
            "response": response,
            "timestamp": time.time(),
            "response_time": response_time,
            "file_name": document_data["file_name"],
        }

        message_1 = {
            "from": "user",
            "message": query_text,
            "timestamp": datetime.now().isoformat(),
        }
        message_2 = {
            "from": "assistant",
            "message": response["result"],
            "timestamp": datetime.now().isoformat(),
        }

        documents_collection.update_one(
            {"file_id": file_id},
            {"$push": {"messages": {"$each": [message_1, message_2]}}},
        )

        logs_collection.insert_one(log_entry)

        return {"response": response["result"]}


@api.route("/list-files")
class ListFiles(Resource):
    def get(self):
        files = list(
            documents_collection.find({}, {"_id": 0, "file_name": 1, "file_id": 1})
        )
        return jsonify({"files": files})


@api.route("/load-chat")
class LoadChat(Resource):
    def post(self):
        file_id = request.json["file_id"]
        document_data = documents_collection.find_one({"file_id": file_id})
        if not document_data:
            return {"error": "Document not found"}, 404

        messages = document_data.get("messages", [])
        return jsonify(
            {
                "messages": messages,
                "file_name": document_data["file_name"],
                "file_id": file_id,
            }
        )


@api.route("/query-volume")
class QueryVolume(Resource):
    def get(self):
        """Returns the number of queries made per day"""
        pipeline = [
            {
                "$project": {
                    "date": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": {"$toDate": {"$multiply": ["$timestamp", 1000]}}
                        }
                    }
                }
            },
            {
                "$group": {
                    "_id": "$date",
                    "query_count": {"$sum": 1}
                }
            },
            {"$sort": {"_id": 1}}
        ]

        result = list(logs_collection.aggregate(pipeline))
        return jsonify({"query_volume": result})


@api.route("/latency-success")
class LatencySuccess(Resource):
    def get(self):
        """Returns LLM latency and success rate per date"""
        pipeline = [
            {
                "$project": {
                    "date": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": {"$toDate": {"$multiply": ["$timestamp", 1000]}},
                        }
                    },
                    "response_time": 1,
                }
            },
            {
                "$group": {
                    "_id": "$date",
                    "avg_latency": {"$avg": "$response_time"},
                    "total_queries": {"$sum": 1},
                    "successful_queries": {
                        "$sum": {"$cond": [{"$gt": ["$response_time", 0]}, 1, 0]}
                    },
                }
            },
            {"$sort": {"_id": 1}},
        ]

        result = list(logs_collection.aggregate(pipeline))

        response_data = [
            {
                "date": entry["_id"],
                "avg_latency": entry["avg_latency"],
                "success_rate": (
                    (entry["successful_queries"] / entry["total_queries"]) * 100
                    if entry["total_queries"] > 0
                    else 0
                ),
            }
            for entry in result
        ]

        return jsonify(response_data)



@api.route("/top-queries")
class TopQueries(Resource):
    def get(self):
        """Returns the most frequently queried documents/questions"""
        pipeline = [
            {"$group": {"_id": "$query", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 10},
        ]
        result = list(logs_collection.aggregate(pipeline))
        return jsonify({"top_queries": result})


@api.route("/top-documents")
class TopDocuments(Resource):
    def get(self):
        """Returns the most queried documents"""
        pipeline = [
            {"$group": {"_id": "$file_id", "query_count": {"$sum": 1}}},
            {"$sort": {"query_count": -1}},
            {"$limit": 10},
        ]
        result = list(logs_collection.aggregate(pipeline))
        return jsonify({"top_documents": result})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
