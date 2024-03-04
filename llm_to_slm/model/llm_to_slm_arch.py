from abc import ABC, abstractmethod

import torch

from llm_to_slm.model.llm_encoder import LlmEncoder
from llm_to_slm.model.projector import get_projector

class LlmModel:
    def __init__(self, config):
        super(LlmModel, self).__init__(config)

        self.llm_encoder = None
        self.projector = None

        if hasattr(config, "llm_encoder") and hasattr(config, "projector_type"):
            self.initialize_llm_modules(
                config.llm_encoder, 
                config.projector_type, 
                config.n_embd
            )

    def initialize_llm_modules(
        self, 
        model_name: str, 
        projector_type: str, 
        slm_hidden_size: int
    ):
        self.llm_encoder = LlmEncoder(model_name)
        
        self.llm_hidden_size = self.llm_encoder.hidden_size
        self.projector = get_projector(projector_type, self.llm_encoder.hidden_size, slm_hidden_size)

        self.config.projector_type = projector_type
        self.config.llm_encoder = model_name

    def get_llm_encoder(self):
        return getattr(self, 'llm_encoder', None)
    

class Llm2SlmForCasualLM(ABC):
    base_model = ""

    @abstractmethod
    def get_model(self):
        pass

    def get_llm_encoder(self):
        return self.get_model().get_llm_encoder()
    
    def encode_prompt_ids(self, prompt_ids):
        prompt_features = self.get_llm_encoder()(prompt_ids)
        prompt_features = self.get_model().projector(prompt_features)
        return prompt_features
    
    def embed(self, input_ids):
        return self.transformer.wte(input_ids)

    def prepare_inputs(self, slm_input_ids, position_ids, attention_mask, past_key_values, labels, llm_input_ids, inputs):
        if llm_input_ids is None or slm_input_ids.shape[1] == 1:
            if past_key_values is not None and llm_input_ids is not None and slm_input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return slm_input_ids, position_ids, attention_mask, past_key_values, None, labels, inputs
        new_input_embeds = self.embed(slm_input_ids) + self.encode_prompt_ids(llm_input_ids)
        
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, labels, inputs
    
    def create_embed(self, slm_input_ids, llm_input_ids):
        new_input_embeds = self.embed(slm_input_ids) + self.encode_prompt_ids(llm_input_ids)
        return new_input_embeds
