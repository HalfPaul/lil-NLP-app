import requests
import json
payload = {'input': 'When did USA civil war happened?'}
response = requests.post("https://qanda-api52.herokuapp.com/predict",  data=json.dumps(payload))
print(response.text)

