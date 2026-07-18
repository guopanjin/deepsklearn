from pyspark.sql import functions as F
from deepsklearn.config import amazon_beauty_config
from pyspark.sql import DataFrame
from pyspark.sql.types import FloatType
from pyspark.sql import Window
import numpy as np
'''
input: user,item_id,title
output:source_item,target_item,score
'''
@F.udf(returnType=FloatType())
def udf_user_hot(user_hot):
    return float(np.round(1 / np.log1p(user_hot), 4))

class Itemcf:
    @classmethod
    def itemcf(cls,spark_raw_data: DataFrame,
               user: str = 'user',
               item_id: str = 'item_id',
               title: str = "title",
               topk: int = 10,
               ):
        df = spark_raw_data.select(user, item_id).drop_duplicates()
        df = df.cache()
        df.count()
        df1 = df.alias("df1")
        df2 = df.alias("df2")
        pairs_df = df1.join(df2, on="user", how="inner").filter(
            F.col("df1.item_id") > F.col("df2.item_id")
        ).select(
            "user",
            F.col("df1.item_id").alias("item_id_1"),
            F.col("df2.item_id").alias("item_id_2")
        )
        # get the popularity of user
        user_hot_df = df.groupby("user").agg(F.count("*").alias("user_hot"))
        pairs_df = pairs_df.join(user_hot_df, how="left", on="user").select(
            "user",
            "item_id_1",
            "item_id_2",
            udf_user_hot(F.col("user_hot")).alias("user_hot_score"))
        concurence_score_df = pairs_df.groupby("item_id_1", "item_id_2").agg(
            F.sum("user_hot_score").alias("concurence_score"))

        item_hot_df = df.groupby("item_id").agg(F.count("*").alias("item_hot"))  # item_id,item_hot

        concurence_score_hot_df = concurence_score_df.join(item_hot_df,
                                                           concurence_score_df["item_id_1"] == item_hot_df["item_id"],
                                                           how="left").select(
            "item_id_1",
            "item_id_2",
            "concurence_score",
            F.col("item_hot").alias("item_hot_1")
        )
        results_df = concurence_score_hot_df.join(item_hot_df,
                                                  concurence_score_hot_df["item_id_2"] == item_hot_df["item_id"],
                                                  how="left").select(
            "item_id_1",
            "item_id_2",
            "concurence_score",
            "item_hot_1",
            F.col("item_hot").alias("item_hot_2")
        ).withColumn("final_score", F.col("concurence_score") / (F.sqrt("item_hot_1") * F.sqrt("item_hot_2"))) \
            .withColumn("final_score", F.round("final_score", 4))

        results_df1 = results_df.select(
            F.col("item_id_1").alias("source_item"),
            F.col("item_id_2").alias("target_item"),
            "concurence_score",
            F.col("item_hot_1").alias("source_item_hot"),
            F.col("item_hot_2").alias("target_item_hot"),
            "final_score"
        )

        results_df2 = results_df.select(
            F.col("item_id_2").alias("source_item"),
            F.col("item_id_1").alias("target_item"),
            "concurence_score",
            F.col("item_hot_2").alias("source_item_hot"),
            F.col("item_hot_1").alias("target_item_hot"),
            "final_score"
        )

        results_df_all = results_df1.unionByName(results_df2)

        w = Window.partitionBy("source_item").orderBy(F.desc("final_score"))
        topk_result = results_df_all.withColumn("row_number", F.row_number().over(w)).filter(
            F.col("row_number") <= topk)
        print(topk_result.count())
        topk_result.show(3)
        return topk_result



