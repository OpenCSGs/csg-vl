from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from .phi import PhiModel, PhiConfig, PhiForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..csg_vl_arch import CSGMetaModel, CSGMetaForCausalLM


class CSGPhiConfig(PhiConfig):
    model_type = "csg-vl-phi"


class CSGPhiModel(CSGMetaModel, PhiModel):
    config_class = CSGPhiConfig

    def __init__(self, config: PhiConfig):
        super(CSGPhiModel, self).__init__(config)


class CSGPhiForCausalLM(PhiForCausalLM, CSGMetaForCausalLM):
    config_class = CSGPhiConfig

    def __init__(self, config):
        super(PhiForCausalLM, self).__init__(config)
        self.model = CSGPhiModel(config)
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
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, # [4, 375]
                position_ids, # None
                attention_mask, # [4, 375]
                past_key_values,
                labels, # [4, 375]
                images # [4, 3, 384, 384]
            )

        return super().forward(
            input_ids=input_ids, # None
            attention_mask=attention_mask,
            position_ids=position_ids, # None
            past_key_values=past_key_values, # None
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache, # None
            output_attentions=output_attentions, # None
            output_hidden_states=output_hidden_states, # None
            return_dict=return_dict # None
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None,
                                      **kwargs):
        images = kwargs.pop("images", None)

        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            **kwargs
        )

        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("csg-vl-phi", CSGPhiConfig)
AutoModelForCausalLM.register(CSGPhiConfig, CSGPhiForCausalLM)
