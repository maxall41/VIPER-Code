from transformers import AutoModel, AutoTokenizer
import torch

def compute_molformer_emb(smiles):
    model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True,add_pooling_layer=True) # ,state_dict=new_state_dict
    tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)
    inputs = tokenizer([smiles], padding=True, return_tensors="pt")
    model.cuda()
    with torch.no_grad():
        inputs.to('cuda')
        outputs = model(**inputs)
        outputs = outputs.pooler_output
    return outputs
