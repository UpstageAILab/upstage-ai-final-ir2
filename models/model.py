from transformers import BertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from models.encoder_model import BertModel


import torch.nn as nn

class SimBERT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        
    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if not return_dict:
            return outputs
    
        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output= outputs,
            last_hidden_state= outputs.last_hidden_state,
            hidden_states= outputs.hidden_states,
        )
            