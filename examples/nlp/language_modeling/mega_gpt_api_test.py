import requests

url = "http://localhost:9000/megagpt"

payload = """{
    "prompt": "What is the meaning of life? A:",
    "tokens_to_generate": 5,
    "stop_after_sentence": "True"
}"""
headers = {
    "Content-Type": "application/json"
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text.encode('utf8'))