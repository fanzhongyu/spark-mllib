package com.timamllib.arithmetic.collaborativefiltering

import org.apache.spark.SparkConf
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.{Row, SparkSession}

/*
spark-mllib ALS参数:
    numBlocks 是用户和项目将被分区以便并行化计算的块数（默认值为10）。
    rank 是模型中潜在因素的数量（默认为10）。
    maxIter 是要运行的最大迭代次数（默认为10）。
    implicitPrefs 指定是使用显式反馈ALS的版本还是用适用于隐式反馈数据集的版本（默认值为 false，这意味着使用显式反馈）。
    alpha 是适用于ALS的隐式反馈版本的参数，用于控制偏好观察值的基线置信度（默认为1.0）。
    nonnegative 指定是否对最小二乘使用非负约束（默认为 false ）。


 */
object CollaborativeFiltering {
  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
  def main(args: Array[String]): Unit = {

    val sparkConf = new SparkConf().setMaster("local").setAppName("CollaborativeFiltering")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    import spark.implicits._

    def parseRating(str: String): Rating = {
      val fields = str.split("::")
      assert(fields.size == 4)
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
    }

    val ratings = spark.read.textFile("data/mllib/sample_movielens_ratings.txt")
      .map(parseRating)
      .toDF()

    ratings.show()
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
    val model = als.fit(training)

    // Evaluate the model by computing the RMSE on the test data

//    model.transform(test)
//      .select("userId", "movieId", "rating")
//      .collect()//返回所有selected数据 in an array
//      .foreach { case Row(userId: Int, movieId: Int, rating: Float) =>
//      println(s"用户：$userId, 电影：$movieId) ->rating：$rating")//自定义输出数据
//    }
    val predictions = model.transform(test)

    predictions.printSchema()
    predictions.show(false)


    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")
  }
}
