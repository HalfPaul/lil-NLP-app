import torch
import torch.nn as nn
import nltk
import json
import random

import bentoml
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.adapters import JsonInput, JsonOutput


@artifacts([bentoml.service.artifacts.BentoServiceArtifact("/json_artifact")])
class MyMLService(BentoService):
    pass