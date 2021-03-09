import torch
from main import QandA




def saveBento():

    bento_svc = QandA()
    bento_svc.pack('onnx_model', './john_model_fp16.onnx')
    bento_svc.save()

if __name__ == '__main__':
    
    saveBento()