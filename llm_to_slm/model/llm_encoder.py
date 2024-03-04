import torch
from torch import nn
from transformers import T5EncoderModel


class LlmEncoder(nn.Module):
    def __init__(
        self, 
        encoder_name: str,
        requires_grad: bool=False
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.requires_grad = requires_grad

        self.load_model()

    def load_model(self):
        if "t5" in self.encoder_name:
            self.encoder = T5EncoderModel.from_pretrained(self.encoder_name)
        else:
            raise ValueError(self.encoder_name)
        self.encoder.requires_grad_(self.requires_grad)

    @torch.no_grad()
    def forward(self, prompt_ids):
        #print(prompt_ids)
        prompt_forward_outs = self.encoder(prompt_ids.to(device=self.device), output_hidden_states=True)

        return prompt_forward_outs.last_hidden_state
    
    @property
    def dtype(self):
        return self.encoder.dtype

    @property
    def device(self):
        return self.encoder.device

    @property
    def config(self):
        return self.encoder.config

    @property
    def hidden_size(self):
        return self.config.hidden_size