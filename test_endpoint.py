import requests
import json

url = "http://127.0.0.1:8000/chat"
headers = {"Content-Type": "application/json"}
data = {"prompt": "What is the capital of France?", "num_samples": 5}

try:
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        print(f"Server returned Error {response.status_code}")
        print(response.text)
    else:
        print("Success!")
        print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Connection failed: {e}")
