# Chatbot-application
Project where I train, prepare and deploy 3 different NLP tasks.

## Introduction
In this project I trained 3 different models:
1. John - Bot that answers your questions using wikipedia and distilled bert model.
2. Sam - Simple conversational chatbot.
3. Fiona - Bot that translates german sentences to english.

## Training John
John was trained using Hugging face library on squad dataset with distilbert model, later I converted model to onnx and quantized it using (**python -m onnxruntime.transformers.optimizer --input model.onnx --output model_fp16.onnx --float16**) command. Distilbert model is a bert model which size was reduced by 40% by using method called distillation(you can read about distillation [here](https://towardsdatascience.com/knowledge-distillation-simplified-dd4973dbc764)).
