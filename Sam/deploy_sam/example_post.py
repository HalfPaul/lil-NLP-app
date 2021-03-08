import requests
import json
payload = {'sentence': 'hello'}
response = requests.post("https://chatbot-api52.herokuapp.com/predict",  data=json.dumps(payload))
print(response.text)