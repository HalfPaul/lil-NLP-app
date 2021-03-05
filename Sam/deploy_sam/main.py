import torch
import torch.nn as nn
import nltk
from nltk.stem.porter import PorterStemmer
import random

import bentoml
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.common import JSONArtifact
from bentoml.adapters import JsonInput

device = torch.device('cpu')



stemmer = PorterStemmer()


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return torch.tensor(bag, dtype=torch.float, device=device)

@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([PytorchModelArtifact('model'), JSONArtifact('intents'), JSONArtifact('data')])
class Chatbot(bentoml.BentoService):
    @bentoml.api(input=JsonInput())
    def predict(self, words):
        words = words["sentence"]   
        all_words = self.artifacts.data["all_words"]
        tags = self.artifacts.data["tags"]
        intents = self.artifacts.intents
        outputs = self.artifacts.model(bag_of_words(words, all_words))
        predicted = torch.argmax(outputs)
        tag = tags[predicted.item()]
        for tg in intents["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        answer = random.choice(responses)
        return answer

