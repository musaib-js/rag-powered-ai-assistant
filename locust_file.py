import os
import random
import uuid
from locust import HttpUser, task, between

class UploadTestUser(HttpUser):
    wait_time = between(0.1, 0.2)  # Keep it low to maintain 10 RPS

    @task
    def upload_file(self):
        file_type = random.choice(["pdf", "csv", "txt"])
        file_content = b"Sample data for testing."  # Binary content for valid PDF
        
        file_name = f"test_file_{uuid.uuid4()}.{file_type}"
        file_path = os.path.join("/tmp", file_name)  # Store temp file

        # Open in binary write mode to prevent corruption
        with open(file_path, "wb") as f:
            f.write(file_content)

        with open(file_path, "rb") as f:
            files = {"file": (file_name, f, "application/octet-stream")}
            response = self.client.post("/upload", files=files)

        os.remove(file_path)  

        assert response.status_code == 200, f"Upload failed: {response.text}"
        assert "document_id" in response.json(), "document_id missing in response"

# Run Locust with:
# locust -f locust_file.py --host=http://localhost:5001 --users 10 --spawn-rate 10
