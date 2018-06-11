package com.timamllib.feature.extractors

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.SparkSession

/**
  * CountVectorizer和CountVectorizerModel旨在帮助将文本文档集合转换为标记数的向量。
  * 当先验词典不可用时，CountVectorizer可以用作估计器来提取词汇表，并生成CountVectorizerModel。
  * 该模型通过词汇生成文档的稀疏表示，然后可以将其传递给其他算法，如LDA。
  */

object CountVectorizer {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("CountVectorizer")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    val df = spark.createDataFrame(Seq(
      (0, Array("a", "b", "c")),
      (1, Array("a", "b", "b", "c", "a"))
    )).toDF("id", "words")

    // fit a CountVectorizerModel from the corpus
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("features")
      .setVocabSize(3)
      .setMinDF(2)
      .fit(df)

    // alternatively, define CountVectorizerModel with a-priori vocabulary
    val cvm = new CountVectorizerModel(Array("a", "b", "c"))
      .setInputCol("words")
      .setOutputCol("features")

    cvModel.transform(df).show()
    cvm.transform(df).show()
  }
}
