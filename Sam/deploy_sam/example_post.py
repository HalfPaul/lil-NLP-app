import requests
import json
payload = {'sentence': 'hello'}
response = requests.post("http://127.0.0.1:5000/predict",  data=json.dumps(payload))
print(response.text)