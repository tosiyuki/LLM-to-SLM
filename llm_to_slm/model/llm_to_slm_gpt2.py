from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, \
                         GPT2LMHeadModel, GPT2Config, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from llm_to_slm.model.llm_to_slm_arch import Llm2SlmForCasualLM, LlmModel


class LLM2SLMConfig(GPT2Config):
    model_type = "llm-to-slm"
        

class Llm2SlmGpt2Model(LlmModel, PreTrainedModel):
    config_class = LLM2SLMConfig

    def __init__(self, config: GPT2Config):
        super(Llm2SlmGpt2Model, self).__init__(config)


class Llm2SlmGpt2ForCausalLM(GPT2LMHeadModel, Llm2SlmForCasualLM):
    config_class = LLM2SLMConfig
    base_model = "gpt2"
    
    def __init__(self, config):
        super(Llm2SlmGpt2ForCausalLM, self).__init__(config)
        self.config = config
        self.model = Llm2SlmGpt2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model 

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        llm_input_ids: Optional[torch.LongTensor] = None,
        inputs: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        print("======================")
        print(f"llm_input_ids: {type(llm_input_ids)}")
        print(f"input_ids: {type(input_ids)}")
        print(f"input_ids: {input_ids.shape}")
        print(f"inputs: {type(inputs)}")
        """
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                inputs
            ) = self.prepare_inputs(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                llm_input_ids,
                inputs
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        llm_input_ids = kwargs.pop("llm_input_ids", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if llm_input_ids is not None:
            _inputs['llm_input_ids'] = llm_input_ids
        return _inputs

AutoConfig.register("llm-to-slm", LLM2SLMConfig)
AutoModelForCausalLM.register(LLM2SLMConfig, Llm2SlmGpt2ForCausalLM)        