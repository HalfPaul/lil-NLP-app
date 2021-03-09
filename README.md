# Chatbot-application
Project where I train, prepare and deploy 2 different NLP tasks. Project is deployed [here](https://lil-nlp-app.herokuapp.com/).

## Introduction
In this project I trained 2 different models:
1. John - Bot that answers your questions using wikipedia and distilled bert model.
2. Sam - Simple conversational chatbot.

## Training John
John was trained using Hugging face library on squad dataset with distilbert model, later I converted model to onnx and quantized it using (**python -m onnxruntime.transformers.optimizer --input model.onnx --output model_fp16.onnx --float16**) command. Distilbert model is a bert model which size was reduced by 40% by using method called distillation(you can read about distillation [here](https://towardsdatascience.com/knowledge-distillation-simplified-dd4973dbc764)).

## Training Sam
Sam was trained on simple pytorch fully connected neural network and intents.json file was passed as data. 

## Deployment
Sam and John were deployed as separate REST APIs on heroku, I used Bentoml as deployment library, Bentoml is a little hard to get used to at first, but once you're comfortable it's very useful library because it handles things like pip requirements and dockerization for you. Front end was built using Flask and also deployed on heroku.

## Acknowledgements
* I used code snippets from [this blog post](https://codinginfinite.com/chatbot-in-python-flask-tutorial/) for chatbot's frontend.
* I used [Python Engineer's](https://www.youtube.com/watch?v=RpWeNzfSUHw) Youtube videos for training the chatbot. 

![Front end design](https://github.com/handertolium/Chatbot-application/blob/main/Readme_imgs/home_photo.png)
