from pyspark.sql import SparkSession
from deepsklearn.config import amazon_beauty_config
from deepsklearn.models import  Itemcf

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

Itemcf.itemcf(spark_raw_data)
