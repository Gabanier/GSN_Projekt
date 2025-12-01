import torch
import torch.nn as nn
from typing import List
from collections import OrderedDict
import yaml

def _to_bool(val):
    return str(val).lower() in ("true", "1", "t", "yes", "on")


def sequential_from_descriptor(file_path):            
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Access the data
    model = data["model"]
    architecture = model["architecture"]
    model_layers = OrderedDict()

    layer_counter = {
        "Linear": 0, "ReLU": 0, "GELU": 0, "SiLU": 0
    }
    for module in architecture:
        layer = module["layer"]
        layer_to_add = None
        type = layer.get("type",None)
        if not type:
            raise Exception("NN layer type not specifed/found")
        if type == "Linear":
            layer_to_add = nn.Linear(in_features=layer["in_features"],
                                     out_features=layer["out_features"],
                                     bias=layer.get("bias",False),
                                     device=layer.get("device",None),
                                     dtype=layer.get("dtype",None))
        elif type == "ReLU":
            layer_to_add = nn.ReLU(inplace=layer.get("inplace",False))

        elif type == "SiLU":
            layer_to_add = nn.SiLU(inplace=layer.get("inplace",False))

        elif type == "GELU":
            layer_to_add = nn.GELU(approximate=layer.get("approximate",'none'))

        if layer_to_add:
            model_layers[f"{type}_{layer_counter[type]}"] = layer_to_add
            layer_counter[type] += 1
    
    return nn.Sequential(model_layers)




if __name__ == "__main__":
    file_path:str = "./config/pinc_model.yaml"
    layers = sequential_from_descriptor(file_path=file_path)
    print(layers)
    
    