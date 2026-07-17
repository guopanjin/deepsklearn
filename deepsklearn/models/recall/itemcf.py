from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from deepsklearn.config import amazon_beauty_config
from pyspark.sql import DataFrame
from pyspark.sql.types import FloatType
from pyspark.sql import Window
import numpy as np
import os

data_path = amazon_beauty_config.itemcf_data_path
spark = (
    SparkSession.builder
        .appName("deepsklearn-aliexpress-profile")
        .master("local[8]")
        .config("spark.driver.memory", "16g")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate())

spark_raw_data = spark.read.parquet(os.path.expanduser(data_path))

spark_raw_data.limit(3).show()
'''
input: user,item_id,title
'''
@F.udf(returnType=FloatType())
def udf_user_hot(user_hot):
    return float(np.round(1 / np.log1p(user_hot), 4))

def itemcf(spark_df: DataFrame,
           user_id_name: str = 'user',
           item_id_name: str = 'item_id',
           item_title: str = "title",
           topk:int=10,
           ):
    df = spark_df.select(user_id_name, item_id_name)
    df=df.cache()
    df.count()
    df1 = df.alias("df1")
    df2=df.alias("df2")
    pairs_df = df1.join(df2, how="inner", on=user_id_name).filter(
        F.col(f"df1.{item_id_name}") > F.col(f"df2.{item_id_name}")).select(
        F.col(f"df1.{user_id_name}"),
        F.col(f"df1.{item_id_name}").alias("item_id_1"),
        F.col(f"df2.{item_id_name}").alias("item_id_2")
    )
    # get the popularity of item_id
    item_hot_df = df1.groupby(item_id_name).agg(F.count("*").alias("item_hot"))  # item_id,item_hot
    # get the popularity of user
    user_hot_df = df1.groupby(user_id_name).agg(F.count("*").alias("user_hot"))
    pairs_df=pairs_df.join(user_hot_df, how="left", on=user_id_name).select(
        user_id_name,
        "item_id_1",
        "item_id_2",
        udf_user_hot(F.col("user_hot")).alias("user_hot_score"))
    pairs_df.show(10)
    concurence_score_df=pairs_df.groupby("item_id_1","item_id_2").agg(F.sum("user_hot_score").alias("concurence_score"))
    concurence_score_hot_df=concurence_score_df.join(item_hot_df,concurence_score_df["item_id_1"]==item_hot_df[item_id_name],how="left").select(
        "item_id_1",
        "item_id_2",
        "concurence_score",
        F.col("item_hot").alias("item_hot_1")
    )
    results_df=concurence_score_hot_df.join(item_hot_df,concurence_score_hot_df["item_id_2"]==item_hot_df[item_id_name],how="left").select(
        "item_id_1",
        "item_id_2",
        "concurence_score",
        "item_hot_1",
        F.col("item_hot").alias("item_hot_2")
    ).withColumn("final_score",F.col("concurence_score")/(F.sqrt("item_hot_1")*F.sqrt("item_hot_2")))
    results_df.show()

    # add title
    title_df=spark_df.select(item_id_name,item_title).drop_duplicates()
    results_title_df1=results_df.join(title_df,results_df["item_id_1"]==title_df[item_id_name],how="left").select(
        "item_id_1",
        title_df[item_title].alias("title_1"),
        "item_id_2",
        "concurence_score",
        "item_hot_1",
        "item_hot_2",
        "final_score"
    )
    results_title_df2=results_title_df1.join(title_df,results_title_df1["item_id_2"]==title_df[item_id_name],how="left").select(
        "item_id_1",
        "title_1",
        "item_id_2",
        title_df[item_title].alias("title_2"),
        "concurence_score",
        "item_hot_1",
        "item_hot_2",
        "final_score"
    )
    results_title_df2.show()

    w1=Window.partitionBy("item_id_2").orderBy(F.desc("final_score"))
    w2= Window.partitionBy("item_id_1").orderBy(F.desc("final_score"))
    topk_df1=results_df.withColumn("row_number",F.row_number().over(w1)).filter(F.col("row_number")<=topk)\

itemcf(spark_raw_data)
