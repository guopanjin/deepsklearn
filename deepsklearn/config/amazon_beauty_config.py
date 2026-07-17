'''
beauty_train_df:262826

beauty_validation_df:44726
n_items:12101

'''

train_dataset="~/.deepsklearn/data/amazon_reviews_beauty_5_2014/beauty_train_dataset.parquet"


validation_dataset="~/.deepsklearn/data/amazon_reviews_beauty_5_2014/beauty_validation_dataset.parquet"


n_items=12101


feature_columns=["user_id","hist_sequence","hist_mask","candidate_id"]

label_columns=["label"]

'''
item_sequence=x["item_sequence"]
padding_mask=x["padding_mask"]
'''

generative_train_data="~/.deepsklearn/data/amazon_reviews_beauty_5_2014/beauty_generative_train_dataset.parquet"
generative_validation_data="~/.deepsklearn/data/amazon_reviews_beauty_5_2014/beauty_generative_validation_dataset.parquet"

generative_feature_columns=["user_index","item_sequence","padding_mask"]
generative_sequence_columns=["item_sequence","padding_mask"]
generative_label_column= "item_sequence"

generative_seq_len=50

###for item2vec
item2vec_train_parquet_path="~/.deepsklearn/data/amazon_reviews_beauty_5_2014/beauty_item2vec_train_dataset.parquet"
item2vec_validation_parquet_path="~/.deepsklearn/data/amazon_reviews_beauty_5_2014/beauty_item2vec_validation_dataset.parquet"
item2vec_sampler_weights_path="~/.deepsklearn/data/amazon_reviews_beauty_5_2014/beauty_item2vec_sampler_weights.parquet"
item2vec_feature_item="feature"
item2vec_label_item="label"

## data for itemcf and swing
itemcf_data_path="~/.deepsklearn/data/amazon_reviews_beauty_5_2014/itemcf.parquet"
