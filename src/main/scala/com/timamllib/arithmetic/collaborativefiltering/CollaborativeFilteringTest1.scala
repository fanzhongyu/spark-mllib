package com.timamllib.arithmetic.collaborativefiltering

import org.apache.spark.SparkConf
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession

/*
基于模型的协同过滤
 */
object CollaborativeFilteringTest1 {
  case class Rating(user: Int, product: Int, rating: Float)
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local").setAppName("CollaborativeFiltering")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    spark.sparkContext.setCheckpointDir("checkpoint")

    import spark.implicits._

    //构造训练数据，格式为dataframe
    val trains = spark.read.textFile("data/mllib/test.data")
      .map(x=>x.split(","))
      .map(x=>Rating(x(0).toInt, x(1).toInt, x(2).toFloat))
      .toDF("user","product","rating")

    trains.show()
    val rank = 10
    val numIterations = 20

    //创建算法实例
    val als = new ALS()
      .setMaxIter(numIterations)
      .setRank(rank)
      .setUserCol("user")
      .setItemCol("product")
      .setRatingCol("rating")

    //ALS Model
    val model = als.fit(trains)

    val rs = model.transform(trains.select("user","product"))
    rs.show()


    //以用户的角度输出
    val userRs = model.recommendForAllUsers(numIterations)


    //以物品的角度输出
    val itemRs = model.recommendForAllItems(numIterations)

    userRs.show(false)

    itemRs.show(false)
  }
}
