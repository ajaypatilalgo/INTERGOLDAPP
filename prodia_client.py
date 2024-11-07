import requests
import time
import json
import base64
from PIL import Image
from io import BytesIO


class Prodia:
    def __init__(self, api_key, base=None):
        self.base = base or "https://api.prodia.com/v1"
        self.headers = {
            "X-Prodia-Key": api_key
        }

    # Generate image using text prompt
    def generate(self, params):
        response = self._post(f"{self.base}/sd/generate", params)
        return response.json()

    # Transform image using img2img parameters
    def transform(self, params):
        response = self._post(f"{self.base}/sd/transform", params)
        return response.json()

    # Get job status by job ID
    def get_job(self, job_id):
        response = self._get(f"{self.base}/job/{job_id}")
        return response.json()

    # Wait for job to complete (succeed or fail)
    def wait(self, job):
        job_result = job
        while job_result['status'] not in ['succeeded', 'failed']:
            time.sleep(0.25)
            job_result = self.get_job(job['job'])
        return job_result

    # List available models
    def list_models(self):
        response = self._get(f"{self.base}/sd/models")
        return response.json()

    # List available samplers
    def list_samplers(self):
        response = self._get(f"{self.base}/sd/samplers")
        return response.json()

    # Helper method to handle POST requests
    def _post(self, url, params):
        headers = {
            **self.headers,
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, data=json.dumps(params))

        if response.status_code != 200:
            print("Error Details:", response.text)
            raise Exception(f"Bad Prodia Response: {response.status_code}")

        return response

    # Helper method to handle GET requests
    def _get(self, url):
        response = requests.get(url, headers=self.headers)

        if response.status_code != 200:
            raise Exception(f"Bad Prodia Response: {response.status_code}")

        return response


# Convert PIL image to Base64 string
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('utf-8')

# Validate input parameters
def validate_params(params):
    for key, value in params.items():
        if value is None or value == '':
            raise ValueError(f"Parameter '{key}' is missing or invalid.")