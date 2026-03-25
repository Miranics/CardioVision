import os

from locust import HttpUser, between, task


class CardioVisionUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(1)
    def health(self):
        self.client.get("/health", name="health")

    @task(3)
    def predict(self):
        sample_path = os.getenv("LOCUST_SAMPLE_IMAGE", "")
        if not sample_path or not os.path.exists(sample_path):
            return

        with open(sample_path, "rb") as handle:
            files = {"file": (os.path.basename(sample_path), handle, "image/jpeg")}
            self.client.post("/predict", files=files, name="predict")
