import random

from locust import HttpUser, between, task

SAMPLE_REVIEWS = [
    "المنتج ممتاز جدا وسريع التوصيل",
    "التجربة كانت سيئة للغاية والمنتج غير مطابق للمواصفات",
    "التطبيق جيد نوعا ما ولكن يتطلب بعض التحسينات",
    "جودة عالية وتغليف ممتاز شكرا لكم",
    "لا انصح بشرائه الخدمة بطيئة جدا",
]


class SentimentAPIUser(HttpUser):
    # Short wait time to simulate concurrent batching load
    wait_time = between(0.1, 0.5)

    @task
    def predict_sentiment(self):
        text = random.choice(SAMPLE_REVIEWS)
        headers = {"Content-Type": "application/json"}
        # Pass text inside a list to match text: list[str] schema
        payload = {"text": [text]}

        self.client.post("/predict", json=payload, headers=headers)