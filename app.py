from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_openai import OpenAI
import os
import uuid
from dotenv import load_dotenv
from flask_restx import Api, Resource
from request_validators import register_models
import pymongo
from doc_loaders import load_documents
from pinecone_vec import create_vector_store, query_vector_store
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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

upload_parser, upload_model, query_model = register_models(api)

llm = OpenAI()

@api.route("/upload")
class Upload(Resource):
    def post(self):
        try:
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

            # Store metadata in MongoDB
            document_data = {
                "file_name": file.filename,
                "file_type": file_type,
                "document_id": document_id,
                "messages": [],
            }
            documents_collection.insert_one(document_data)

            documents = load_documents(file_path, file_type)
            create_vector_store(documents, document_id)

            os.remove(file_path)

            return jsonify(
                {
                    "message": "File uploaded and indexed successfully",
                    "document_id": document_id,
                }
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)


@api.route("/query")
class Query(Resource):
    @api.expect(query_model)
    def post(self):
        file_id = request.json["file_id"]
        query_text = request.json["query"]

        document_data = documents_collection.find_one({"file_id": file_id})
        if not document_data:
            return {"error": "Document not found"}, 404

        matches = query_vector_store(document_data["document_id"], query_text)

        if not matches:
            return {"response": "No relevant documents found."}

        # Concatenate top chunks as context
        context = "\n\n".join([match["metadata"]["text"] for match in matches])

        # Ask LLM
        start_time = time.time()
        prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query_text}"
        result = llm.invoke(prompt)
        response_time = time.time() - start_time

        # Store messages & logs
        message_1 = {
            "from": "user",
            "message": query_text,
            "timestamp": datetime.now().isoformat(),
        }
        message_2 = {
            "from": "assistant",
            "message": result,
            "timestamp": datetime.now().isoformat(),
        }

        documents_collection.update_one(
            {"file_id": file_id},
            {"$push": {"messages": {"$each": [message_1, message_2]}}},
        )

        logs_collection.insert_one({
            "file_id": file_id,
            "query": query_text,
            "response": {"result": result},
            "timestamp": time.time(),
            "response_time": response_time,
            "file_name": document_data["file_name"],
        })

        return {"response": result}


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
        pipeline = [
            {"$group": {"_id": "$file_id", "query_count": {"$sum": 1}}},
            {"$sort": {"query_count": -1}},
            {"$limit": 10},
        ]
        result = list(logs_collection.aggregate(pipeline))
        return jsonify({"top_documents": result})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
