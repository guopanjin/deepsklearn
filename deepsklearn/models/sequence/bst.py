import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsklearn.layers import EncoderLayer
from deepsklearn.initializations import init_embedding
from deepsklearn.layers import MLPBlock

'''
paper hyper parameter:
 encoderlayer:1 # short sequence does not need multi-encoder
 num_heads:8
 item embedding_dim=32
 embedding_dim for encoder:32*2=64
 max_len:20
 MLP:1024-512-256
 
Behavior Sequence Transformer:
  embedding_matrix=[num_items+1,embed_dim] padding_idx=0
  
  [batch_size,seq_len]-->padding-->[batch_size,max_len] -->lookup embedding
  -->[batch_size,max_len,embed_dim]--->-
  --->add candidate-->[batch_size,max_len+1,embed_dim]--> +position embedding--> [batch_size,max_len+1,embed_dim]
  -->encoderlayer([batch_size,max_len+1,embed_dim],key_padding_mask)   -->[batch_size,max_len+1,embed_dim]
  --->mask-->[batch_size,max_len+1,embed_dim]--->flattern
  --->[batch_size,(max_len+1)*embed_dim]-->mlp--->linear--->logits
'''
class BST(nn.Module):
    def __init__(self,
                 num_items,
                 embed_dim:int=32,
                 max_len:int=50,
                 num_heads=2,
                 hidden_layers=(128,64,32),
                 drop_out=0.0,
                 ffn_dim=128
                 ):
        super().__init__()
        self.num_items=num_items
        self.embed_dim=embed_dim
        self.max_len=max_len
        self.num_heads=num_heads
        self.ffn_dim=ffn_dim
        self.hidden_layers=hidden_layers
        self.drop_out=drop_out
        #padding_idx=0 means this line embedding will not update the gradient
        self.item_embedding_table=nn.Embedding(num_embeddings=self.num_items+1,embedding_dim=self.embed_dim, padding_idx=0)
        self.position_encoding_table=nn.Embedding(num_embeddings=self.max_len+1,embedding_dim=self.embed_dim)
        self.register_buffer("positions",torch.arange(self.max_len+1))
        self.encoder_layer=EncoderLayer(embed_dim=self.embed_dim,
                                        num_heads=self.num_heads,
                                        ffn_dim=self.ffn_dim
                                        )
        self.mlp_block=MLPBlock(input_dim=(self.max_len+1)*self.embed_dim,
                                hidden_layers=self.hidden_layers,
                                dropout=self.drop_out
                                )
        self.output_linear_layer=nn.Linear(self.hidden_layers[-1],1)
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
    def forward(self,x):
        hist_item_ids=x["hist_sequence"]
        batch_size=hist_item_ids.shape[0]
        hist_mask=x["hist_mask"]
        key_padding_mask=(hist_mask==0)
        key_padding_mask=torch.concat([key_padding_mask,torch.full((batch_size,1),False).to(hist_mask.device)],dim=1) #(batch_size,max_len+1)
        candidate_item_id=x["candidate_id"]
        sequence_embedding=self.item_embedding_table(hist_item_ids)#(batch_size,max_len,embed_dim)
        candidate_embedding=self.item_embedding_table(candidate_item_id)#(batch_size,embed_dim)
        candidate_embedding=torch.unsqueeze(candidate_embedding,dim=1)#(batch_size,1,embed_dim)
        encoder_input=torch.concat([sequence_embedding,candidate_embedding],dim=1) #(batch_size,max_len+1,embed_dim)
        position_embedding=self.position_encoding_table(self.positions)
        encoder_input=encoder_input+position_embedding
        encoder_output=self.encoder_layer(x=encoder_input,key_padding_mask=key_padding_mask) #(batch_size,max_len+1,embed_dim)
        key_padding_mask=torch.unsqueeze(key_padding_mask, dim=-1) #(batch_size,max_len+1,embed_dim,1)
        encoder_output=torch.masked_fill(encoder_output,key_padding_mask,float(0.0))

        mlp_input=torch.flatten(encoder_output,start_dim=1,end_dim=-1)
        mlp_out=self.mlp_block(mlp_input)
        logits=self.output_linear_layer(mlp_out)
        return logits