package com.timamllib.arithmetic.classification

import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._

object LogisticRegressionEmail {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local").setAppName("LogisticRegressionEmail")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    import spark.implicits._

    val trainingSpam = spark.sparkContext.textFile("data/mllib/email/spam.txt").toDF("email").withColumn("label",lit(1))
    val trainingNormal = spark.sparkContext.textFile("data/mllib/email/normal.txt").toDF("email").withColumn("label",lit(0))
    val train = trainingSpam.union(trainingNormal)
    train.show(false)

    // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    val tokenizer = new Tokenizer()
      .setInputCol("email")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setNumFeatures(100)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)
      .setFeaturesCol("features")
      .setLabelCol("label")
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr))

    // Fit the pipeline to training documents.
    val model = pipeline.fit(train)

    val testData = spark.createDataFrame(Seq(
        (1L,"O M G GET cheap stuff by sending money to ...",1),
        (2L,"Hi Dad, I started studying Spark the other ...",0),
        (3L,"I really wish well to all my friends.",1),
        (4L,"He stretched into his pocket for some money.",1),
        (5L,"He entrusted his money to me.",1),
        (6L,"Where do you keep your money?",1),
        (7L,"She borrowed some money of me.",1)
      )).toDF("id","email","result")

    // Make predictions on test documents.
    model.transform(testData)
      .select("id", "email", "probability", "prediction","result")
      .collect()
      .foreach { case Row(id: Long, email: String, prob, prediction: Double,result:Int) =>
        println(s"($id, $email) --> prob=$prob, 预测结果为=$prediction 实际结果为=$result")
      }
  }
}
