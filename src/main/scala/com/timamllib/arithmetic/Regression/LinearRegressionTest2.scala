package com.timamllib.arithmetic.Regression

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.{DataFrame, SparkSession}

// Population人口,
// Income收入水平,
// Illiteracy文盲率,
// LifeExp,
// Murder谋杀率,
// HSGrad,
// Frost结霜天数(温度在冰点以下的平均天数) ,
// Area州面积
object LinearRegressionTest2 {

  case class dataStructure(Population:Int,Income:Double, Illiteracy:Double,
                           LifeExp:Double, Murder:Double, HSGrad:Double, Frost:Int, Area:Int)

  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local").setAppName("BinomialLogisticRegression")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    import spark.implicits._

    val sourceData = spark.sparkContext.textFile("data/mllib/linear_regression_data.txt")

    val data =sourceData.map(x=>x.split(",")).map(x=>
      dataStructure(x(0).toInt,x(1).toDouble,x(2).toDouble,x(3).toDouble,
        x(4).toDouble,x(5).toDouble,x(6).toInt,x(7).toInt)).toDF()

    val colArray = Array("Population", "Income", "Illiteracy", "LifeExp", "HSGrad", "Frost", "Area")

    val assembler = new VectorAssembler().setInputCols(colArray).setOutputCol("features")

    val vecDF: DataFrame = assembler.transform(data)

    val lr = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("Murder")
      .setFitIntercept(true)
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(vecDF)

    // 输出模型全部参数
    lrModel.extractParamMap()
    // Print the coefficients and intercept for linear regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    val predictions = lrModel.transform(vecDF)
    predictions.printSchema()
    predictions.selectExpr("Murder", "round(prediction,1) as prediction").show

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")


  }
}
