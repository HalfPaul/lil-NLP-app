import torch
import json
from main import Chatbot
import torch
import torch.nn as nn 

device = torch.device('cpu')

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

def saveBento():
    with open('intents.json', 'r') as f:
        intents = json.load(f)

    data = torch.load("data.pth")

    model = torch.load("model.pth")


    bento_svc = Chatbot()
    bento_svc.pack('model', model.to(device))
    bento_svc.pack('intents', intents)
    bento_svc.pack('data', data)
    

    bento_svc.save()

if __name__ == '__main__':
    
    saveBento()
