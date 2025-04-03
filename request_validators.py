from flask_restx import fields, Model, reqparse

def register_models(api):
    """Registers models with the API"""

    # Parser for file uploads
    upload_parser = reqparse.RequestParser()
    upload_parser.add_argument(
        "file",
        type=str,
        location="files",
        required=True,
        help="File to be uploaded (PDF/CSV/Text)",
    )

    # Model for API documentation (doesn't work for file uploads directly)
    upload_model = api.model(
        "UploadModel",
        {
            "file": fields.String(description="File to be uploaded (PDF/CSV/Text)"),
        },
    )

    # Model for query requests
    query_model = api.model(
        "QueryModel",
        {
            "query": fields.String(required=True, description="Query to be answered"),
        },
    )

    return upload_parser, upload_model, query_model
