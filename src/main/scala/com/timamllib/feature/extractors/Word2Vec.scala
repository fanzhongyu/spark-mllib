package com.timamllib.feature.extractors

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.{Row, SparkSession}
/*
Word2Vec是一个Estimator(评估器)，它采用表示文档的单词序列，并训练一个Word2VecModel。
该模型将每个单词映射到一个唯一的固定大小向量。Word2VecModel使用文档中所有单词的平均值将每个文档转换为向量;
 该向量然后可用作预测,文档相似性计算等功能。
 */

object Word2Vec {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("Word2Vec")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    // Input data: Each row is a bag of words from a sentence or document.
    /**
    +--------------------+
    |                text|
    +--------------------+
    |[Hi, I, heard, ab...|
    |[I, wish, Java, c...|
    |[Logistic, regres...|
    +--------------------+
     */
    val documentDF = spark.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")

    documentDF.show(100)

    // Learn a mapping from words to Vectors.
    val word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)
    val model = word2Vec.fit(documentDF)

    val result = model.transform(documentDF)

    /**
    Text: [Hi, I, heard, about, Spark] =>
    Vector: [0.03171061240136624,0.009584793448448183,0.024160488322377206]

    Text: [I, wish, Java, could, use, case, classes] =>
    Vector: [0.025792076385446956,0.031522061077079604,-0.018240033016939248]

    Text: [Logistic, regression, models, are, neat] =>
    Vector: [0.022660286724567415,-0.01607915610074997,0.051439096033573155]
    */
    result.collect().foreach { case Row(text: Seq[_], features) =>
      println(s"Text: [${text.mkString(", ")}] => \nVector: $features\n") }
  }
}
