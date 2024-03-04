
import re
from torch import nn


def get_projector(projector_type: str, mm_hidden_size: int, hidden_size: int):
    mlp_relu_match = re.match(r'^mlp(\d+)x_relu$', projector_type)
    if mlp_relu_match:
        mlp_depth = int(mlp_relu_match.group(1))
        modules = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        return nn.Sequential(*modules)

    raise ValueError(f'Unknown projector type: {projector_type}')