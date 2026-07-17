import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsklearn.layers import MLPBlock
from deepsklearn.initializations import init_embedding
'''
input:
 { 
  "user_index":[batch,]
  "item_sequence":[batch,seq_len]
  "padding_mask":[batch,seq_len]
  "label":label_data
}
'''
class TwoTower(nn.Module):
    def __init__(self,
                 num_items:int,
                 embed_dim:int=32,
                 hidden_layers=(128,64,32)
                 ):
        super().__init__()
        self.num_items=num_items
        self.embed_dim=embed_dim
        self.hidden_layers=hidden_layers
        self.item_embedding_table=nn.Embedding(self.num_items+1,self.embed_dim,padding_idx=0)
        self.user_mlp=MLPBlock(input_dim=self.embed_dim,hidden_layers=self.hidden_layers)
        self.item_mlp=MLPBlock(input_dim=self.embed_dim,hidden_layers=self.hidden_layers)
        self.user_linear=nn.Linear(self.hidden_layers[-1],self.embed_dim)
        self.item_linear = nn.Linear(self.hidden_layers[-1], self.embed_dim)
        self.apply(lambda m:init_embedding(m))

    def forward(self,x):
        item_sequence=x["item_sequence"] #(batch_size,seq_len)
        padding_mask=x["padding_mask"] #(batch_size,seq_len)
        valid_seq_len=torch.clamp_min(torch.sum(padding_mask,dim=1,keepdim=True),1)#(batch_size,1)
        padding_mask=(padding_mask==0)
        padding_mask=torch.unsqueeze(padding_mask,dim=-1) #(batch_size,seq_len,1)
        positive_itemid=x["label"] #(batch_size,)
        user_seq_embedding=self.item_embedding_table(item_sequence)#(batch_size,seq_len,embed_dim)
        user_embedding=torch.sum(torch.masked_fill(user_seq_embedding,padding_mask,float(0.0)),dim=1)/valid_seq_len #(batch_size,embed_dim)
        user_output=self.user_mlp(user_embedding)
        user_output=self.user_linear(user_output)

        itemid_embedding=self.item_embedding_table(positive_itemid)#(batch_size,embed_dim)
        item_output=self.item_mlp(itemid_embedding)
        item_output=self.item_linear(item_output)


        logits=user_output@item_output.T#(batch_size,batch_size)
        labels=torch.arange(logits.shape[0],device=logits.device)#(batch_size,)
        return logits,labels