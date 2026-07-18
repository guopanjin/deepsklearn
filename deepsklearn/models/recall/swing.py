from pyspark.sql import DataFrame
from pyspark.sql.types import FloatType
from pyspark.sql import Window
import numpy as np
from pyspark.sql import functions as F

'''
input: user,item_id,title
output:source_item,target_item,
'''
class Swing():
    @classmethod
    def swing(
              cls,
              spark_raw_data: DataFrame,
              user: str = 'user',
              item_id: str = 'item_id',
              title: str = "title",
              topk: int = 10,
              ):
        # step1:get the num_items that each 2 users cliked
        spark_user_item_dis = spark_raw_data.select(user, item_id).dropDuplicates()
        spark_user_item_dis = spark_user_item_dis.cache()
        df1 = spark_user_item_dis.alias("df1")
        df2 = spark_user_item_dis.alias("df2")
        df3 = df1.join(df2, "item_id", how="inner").filter(F.col("df1.user") > F.col("df2.user")).select(
            F.col("df1.user").alias("user_1"),
            F.col("df2.user").alias("user_2"),
            "item_id"
        ).groupby(
            "user_1", "user_2"
        ).agg(F.count("*").alias("co_num_items")) \
            .select(
            F.col("user_1").alias("source_user"),
            F.col("user_2").alias("target_user"),
            "co_num_items")
        # step2: get co-occurrence for item pair
        df1 = spark_user_item_dis.alias("df1")
        df2 = spark_user_item_dis.alias("df2")
        user_item_pair_df = df1.join(df2, "user").filter(F.col("df1.item_id") > F.col("df2.item_id")).select(
            F.col("df1.item_id").alias("item_1"),
            F.col("df2.item_id").alias("item_2"),
            "user"
        )
        pair_df1 = user_item_pair_df.alias("pair_df1")
        pair_df2 = user_item_pair_df.alias("pair_df2")
        pair_df3 = pair_df1.join(pair_df2, ["item_1", "item_2"], how="inner") \
            .filter(F.col("pair_df1.user") > F.col("pair_df2.user")).select(
            "item_1",
            "item_2",
            F.col("pair_df1.user").alias("source_user"),
            F.col("pair_df2.user").alias("target_user")
        )
        pair_df4 = pair_df3.join(df3, ["source_user", "target_user"]).select(
            "item_1",
            "item_2",
            "source_user",
            "target_user",
            "co_num_items")
        # step3:swing results
        swing_result_df = pair_df4.withColumn("pair_score", 1 / (1 + F.col("co_num_items"))) \
            .groupBy("item_1", "item_2") \
            .agg(F.sum("pair_score").alias("sum_paire_score")) \
            .select("item_1",
                    "item_2",
                    F.col("sum_paire_score").alias("score")
                    )
        # step4:get topk
        swing_result_df = swing_result_df.cache()
        swing_result_df.count()
        result_df1 = swing_result_df \
            .select(F.col("item_1").alias("source_item"), F.col("item_2").alias("target_item"), "score")
        result_df2 = swing_result_df \
            .select(F.col("item_2").alias("source_item"), F.col("item_1").alias("target_item"), "score")

        result_df3 = result_df1.unionByName(result_df2).withColumn("score", F.round("score", 4))

        w = Window.partitionBy("source_item").orderBy(F.desc("score"))
        topk = 10

        swing_final_results = result_df3.withColumn("row_number", F.row_number().over(w)) \
            .filter(F.col("row_number") <= topk)
        print(swing_final_results.count())
        swing_final_results.show(3)
        return swing_final_results





