package com.timamllib.feature.transformers

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

/**
  * StringIndexer（字符串-索引变换）将标签的字符串列编号变成标签索引列。
  * 标签索引序列的取值范围是[0，numLabels（字符串中所有出现的单词去掉重复的词后的总和）]，
  * 按照标签出现频率排序，出现最多的标签索引为0。如果输入是数值型，我们先将数值映射到字符串，
  * 再对字符串进行索引化。如果下游的 pipeline（例如：Estimator 或者 Transformer）需要用到索引化后的标签序列，
  * 则需要将这个 pipeline 的输入列名字指定为索引化序列的名字
  */
object StringIndexer {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("StringIndexer")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    import org.apache.spark.ml.feature.StringIndexer

    val df = spark.createDataFrame(
      Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
    ).toDF("id", "category")

    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")

    val indexed = indexer.fit(df).transform(df)
    indexed.show()
  }
}
