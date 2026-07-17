import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsklearn.layers import MLPBlock
from deepsklearn.initializations import init_embedding
'''
input:
 {
    "f1":(batch_size,),
    "f2":(batch_size,),
    "f3":(batch_size,),
 },
 {
   "ctr_label",(batch_size,),
   "ctcvr_label",(batch_size,),
 }
'''
class ESMM(nn.Module):
    def __init__(self,
                 categorical_feature_columns:dict=None,
                 numerical_feature_columns:list=None,
                 embed_dim:int=32,
                 hidden_layer=(128,64,32)
                 ):
        super().__init__()
        self.categorical_feature_columns=categorical_feature_columns
        self.numerical_feature_columns=numerical_feature_columns
        self.embed_dim=embed_dim
        self.hidden_layer=hidden_layer
        embedding_table_dict={feature_name:nn.Embedding(num_embeddings,self.embed_dim) for feature_name, num_embeddings in  self.categorical_feature_columns.items()}
        self.feature_embedding_table_dict=nn.ModuleDict(embedding_table_dict)

        self.ctr_tower=MLPBlock(input_dim=len(categorical_feature_columns)*self.embed_dim+len(numerical_feature_columns),
                           hidden_layers=hidden_layer
                           )
        self.cvr_tower=MLPBlock(input_dim=len(categorical_feature_columns)*self.embed_dim+len(numerical_feature_columns),
                            hidden_layers=hidden_layer
                           )
        self.ctr_out_linear=nn.Linear(hidden_layer[-1],1)
        self.cvr_out_linear = nn.Linear(hidden_layer[-1], 1)
        self.apply(lambda m:init_embedding(m))

    def forward(self,x,label_dict):
        categorical_feature_embeddings=[self.feature_embedding_table_dict[feature_name](x[feature_name]) for feature_name in self.categorical_feature_columns.keys()]
        categorical_feature_embeddings=torch.stack(categorical_feature_embeddings,dim=1) #(batch_size,category_feature_size,embed_dim)

        numerical_feature=[x[feature_name] for feature_name in self.numerical_feature_columns]
        numerical_feature=torch.stack(numerical_feature,dim=-1)#(batch_size,numerical_feature_size)
        categorical_feature=torch.reshape(categorical_feature_embeddings,(categorical_feature_embeddings.shape[0],-1))#(batch_size,category_feature_size*embed_dim)

        feature_input=torch.concat([categorical_feature,numerical_feature],dim=1)#(batch_size,numerical_feature_size+category_feature_size*embed_dim)

        ctr_out=self.ctr_tower(feature_input)
        ctr_logits=self.ctr_out_linear(ctr_out)
        ctr_logits=torch.squeeze(ctr_logits,dim=-1)


        cvr_out = self.cvr_tower(feature_input)
        cvr_logits = self.cvr_out_linear(cvr_out)
        cvr_logits=torch.squeeze(cvr_logits,dim=-1)

        ctr_label = label_dict["click"].to(torch.float)
        ctcvr_label = label_dict["conversion"].to(torch.float)

        ctr_loss=F.binary_cross_entropy_with_logits(ctr_logits,ctr_label)

        pctr=F.sigmoid(ctr_logits)
        pcvr=F.sigmoid(cvr_logits)
        pctcvr=pctr*pcvr
        ctcvr_loss=F.binary_cross_entropy(pctcvr, ctcvr_label)


        loss=ctr_loss+ctcvr_loss

        return {
            "loss": loss,
            "ctr_loss": ctr_loss,
            "ctr_label": ctr_label,
            "ctcvr_loss": ctcvr_loss,
            "ctcvr_label": ctcvr_label,
            "pctr": pctr,
            "pctcvr": pctcvr}






