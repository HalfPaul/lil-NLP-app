import bentoml
from bentoml.frameworks.onnx import OnnxModelArtifact
from bentoml.frameworks.transformers import TransformersModelArtifact
from bentoml.adapters import JsonInput

import wikipedia as wiki
import numpy as np
import torch
from transformers import AutoTokenizer


@bentoml.env(pip_packages=['torch==1.6.0', "wikipedia", "transformers"])
@bentoml.artifacts([OnnxModelArtifact('onnx_model', backend='onnxruntime')])
class QandA(bentoml.BentoService):
    @bentoml.api(input=JsonInput())
    def predict(self, question):
        # Api request will look like {input: question}
        results = wiki.search(question["input"])
        page = wiki.page(results[0])
        context = page.content

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
        tokens = tokenizer.encode_plus(question["input"], context[:5000], return_tensors="pt", max_length=212, truncation=True)
        inputs_onnx = {"input_ids":np.array(tokens["input_ids"][:212], dtype = "int64"), "attention_mask":np.array(tokens["attention_mask"][:212], dtype = "int64")}

        output = self.artifacts.onnx_model.run(None, inputs_onnx)
        answer_start = np.argmax(output[0])  
        answer_end = np.argmax(output[1]) + 1
        output = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokens["input_ids"][0][answer_start:answer_end]))
        return output



 
