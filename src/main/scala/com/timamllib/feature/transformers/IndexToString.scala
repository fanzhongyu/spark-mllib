package com.timamllib.feature.transformers

import org.apache.spark.SparkConf
import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.SparkSession

/**
  *与 StringIndexer 对应，IndexToString 将索引化标签还原成原始字符串。
  * 一个常用的场景是先通过 StringIndexer 产生索引化标签，然后使用索引化标签进行训练，
  * 最后再对预测结果使用 IndexToString来获取其原始的标签字符串。
  */
object IndexToString {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("IndexToString")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    val df = spark.createDataFrame(Seq(
      (0, "a"),
      (1, "b"),
      (2, "c"),
      (3, "a"),
      (4, "a"),
      (5, "c")
    )).toDF("id", "category")

    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df)
    val indexed = indexer.transform(df)

    println(s"Transformed string column '${indexer.getInputCol}' " +
      s"to indexed column '${indexer.getOutputCol}'")
    indexed.show()

    val inputColSchema = indexed.schema(indexer.getOutputCol)
    println(s"StringIndexer will store labels in output column metadata: " +
      s"${Attribute.fromStructField(inputColSchema).toString}\n")

    val converter = new IndexToString()
      .setInputCol("categoryIndex")
      .setOutputCol("originalCategory")

    val converted = converter.transform(indexed)

    println(s"Transformed indexed column '${converter.getInputCol}' back to original string " +
      s"column '${converter.getOutputCol}' using labels in metadata")
    converted.select("id", "categoryIndex", "originalCategory").show()
  }
}
