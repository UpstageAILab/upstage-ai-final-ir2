from transformers import AutoTokenizer, AutoModel

from train.models.model import SimBERT

def get_model(model_name):
    return SimBERT.from_pretrained(model_name)
    
def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)