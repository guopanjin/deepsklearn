import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsklearn.initializations import init_embedding

'''
input: 
    feature_item_ids:(batch_size,)
    positive_item_ids:(batch_size,)
    negative_item_ids:(batch_size,k)

feature_embedding_table=nn.Embedding(num_items+1,embed_dim)
target_embedding_table=nn.Embedding(num_items+1,embed_dim)

feature_item_ids->feature_embedding_table->feature_item_embedding #(batch_size,embed_dim)
positive_item_ids->target_embedding_table->positive_item_embedding #(batch_size,embed_dim)
negative_item_ids->target_embedding_table->negative_item_embedding #(batch_size,k,embed_dim)
feature_item_embedding-->(batch_size,1,embed_dim)-->(batch_size,k,embed_dim)-->netative_feature_item_embedding
positve_logits= torch.sum(feature_item_embedding*positive_item_embedding,dim=-1).reshape(batch_size,1)

negative_logits= torch.sum(netative_feature_item_embedding*negative_item_embedding,dim=-1).reshape(batch_size*k,1)

logits=torch.concat([positve_logits,negative_logits] ,dim=0) #(batch_size*k+batch_size,1)
positive_label=torch.ones((batch_size,1))
negative_label=torch.zeros((batch_size*k,1))
labels=torch.concat([positive_label,negative_label],dim=0)


'''
class Item2vec(nn.Module):
    def __init__(self,
                 num_items: int,
                 embed_dim:int=32
                 ):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_items=num_items
        self.input_embedding_table=nn.Embedding(self.num_items+1,self.embed_dim)
        self.output_embedding_table=nn.Embedding(self.num_items+1,self.embed_dim)
        self.apply(lambda m:init_embedding(m))

    def forward(self,x):
        feature_item=x["features"]#(batch)
        positive_items=x["positive_items"]#(batch,)
        negative_items=x["negative_items"] #(batch,k)
        batch_size,k=negative_items.shape
        feature_item_embedding=self.input_embedding_table(feature_item)
        positive_items_embedding = self.output_embedding_table(positive_items)

        negative_items_embedding= self.output_embedding_table(negative_items)

        positive_logits=torch.sum(feature_item_embedding*positive_items_embedding,dim=1,keepdim=True)
        positive_labels=torch.ones_like(positive_logits)
        negative_feature_embedding=torch.unsqueeze(feature_item_embedding,dim=1).expand((batch_size,k,self.embed_dim))
        negative_logits=torch.sum(negative_feature_embedding*negative_items_embedding,dim=-1).reshape(batch_size*k,1)
        negative_lables=torch.zeros_like(negative_logits)

        logits=torch.concat([positive_logits,negative_logits],dim=0)
        labels=torch.concat([positive_labels,negative_lables],dim=0)
        return logits,labels

