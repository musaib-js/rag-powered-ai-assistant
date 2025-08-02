from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader

def load_documents(file_path, file_type):
    """Loads documents based on their file type."""
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "csv":
        loader = CSVLoader(file_path)
    else:
        loader = TextLoader(file_path)
    return loader.load()

def load_docs_multiple(file_paths):
    """Loads documents based on their file type."""
    
    docs = []

    for file in file_paths:
        if file["file_type"] == "pdf":
            loader = PyPDFLoader(file["file_path"])
        elif file["file_type"]:
            loader = CSVLoader(file["file_path"])
        else:
            loader = TextLoader(file["file_path"])
        doc =  loader.load()
    
    
