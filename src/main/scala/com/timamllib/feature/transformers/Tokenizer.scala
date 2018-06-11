package com.timamllib.feature.transformers

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Tokenizer {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("Tokenizer")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    val sentenceDataFrame = spark.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (1, "I wish Java could use case classes"),
      (2, "Logistic,regression,models,are,neat")
    )).toDF("id", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")
      .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

    val countTokens = udf { (words: Seq[String]) => words.length }

    /**
    +-----------------------------------+------------------------------------------+------+
    |sentence                           |words                                     |tokens|
    +-----------------------------------+------------------------------------------+------+
    |Hi I heard about Spark             |[hi, i, heard, about, spark]              |5     |
    |I wish Java could use case classes |[i, wish, java, could, use, case, classes]|7     |
    |Logistic,regression,models,are,neat|[logistic,regression,models,are,neat]     |1     |
    +-----------------------------------+------------------------------------------+------+
    */
    val tokenized = tokenizer.transform(sentenceDataFrame)
    tokenized.select("sentence", "words")
      .withColumn("tokens", countTokens(col("words"))).show(false)

    //同上
    val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
    regexTokenized.select("sentence", "words")
      .withColumn("tokens", countTokens(col("words"))).show(false)
  }
}
