import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsklearn.layers import EncoderLayer
from deepsklearn.initializations import init_embedding
from deepsklearn.layers import MLPBlock


class AvgPooling(nn.Module):
    def __init__(self,
                 num_items,
                 embed_dim: int = 32,
                 max_len: int = 50,
                 hidden_layers=(128, 64, 32),
                 drop_out=0.0):
        super().__init__()
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.hidden_layers = hidden_layers
        self.drop_out = drop_out
        # padding_idx=0 means this line embedding will not update the gradient
        self.item_embedding_table = nn.Embedding(num_embeddings=self.num_items + 1, embedding_dim=self.embed_dim,
                                                 padding_idx=0)

        self.mlp_block = MLPBlock(input_dim=self.embed_dim*3,
                                  hidden_layers=self.hidden_layers,
                                  dropout=self.drop_out
                                  )
        self.output_linear_layer = nn.Linear(self.hidden_layers[-1], 1)
        self.apply(lambda m: init_embedding(m))

    '''
    {
      "user_id":userid,
      "hist_sequence": [batch_size,max_len]
      "hist_mask":[batch_size,max_len]
      "candidate_id":[batch_size,]
      "label":[batch_size,]
    }
    '''

    def forward(self, x):
        hist_item_ids = x["hist_sequence"]
        hist_mask = x["hist_mask"] #(batch_size,feature_size)
        candidate_item_id = x["candidate_id"]
        seq_len=torch.sum(hist_mask,dim=1,keepdim=True)#(batch_size,1)
        key_padding_mask = (hist_mask == 0) #(batch_size,feature_size)

        hist_embedding=self.item_embedding_table(hist_item_ids)#(batch_size,feature_size,embed_dim)
        hist_embedding=torch.masked_fill(hist_embedding,torch.unsqueeze(key_padding_mask,dim=-1),float(0.0))
        candidate_embedding=self.item_embedding_table(candidate_item_id)#(batch_size,embed_dim)

        user_vector=torch.sum(hist_embedding,dim=1)/(seq_len+1e-9)#(batch_size,embed_dim)

        mlp_input=torch.concat([user_vector,candidate_embedding,user_vector*candidate_embedding],dim=1)
        mlp_out=self.mlp_block(mlp_input)
        logits=self.output_linear_layer(mlp_out)
        return logits