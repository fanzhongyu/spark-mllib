package com.timamllib.arithmetic.classification

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.max

/*
二项逻辑回归
 */
object BinomialLogisticRegression {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local").setAppName("BinomialLogisticRegression")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    import spark.implicits._

    // Load training data
    val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(training)

    // Print the coefficients and intercept for logistic regression
    println(s"BinomialLogisticRegression Coefficients(系数): ${lrModel.coefficients} \nBinomialLogisticRegression Intercept(截距): ${lrModel.intercept}")

    // We can also use the multinomial family for binary classification
    val mlr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFamily("multinomial")

    val mlrModel = mlr.fit(training)

    // Print the coefficients and intercepts for logistic regression with multinomial family
    println(s"Multinomial coefficients(系数): ${mlrModel.coefficientMatrix}")
    println(s"Multinomial intercepts(截距): ${mlrModel.interceptVector}")

    // Extract the summary from the returned LogisticRegressionModel instance trained in the earlier
    // example
    val trainingSummary = lrModel.summary

    // Obtain the objective per iteration.
    val objectiveHistory = trainingSummary.objectiveHistory
    println("objectiveHistory:")
    objectiveHistory.foreach(loss => println(loss))

    // Obtain the metrics useful to judge performance on test data.
    // We cast the summary to a BinaryLogisticRegressionSummary since the problem is a
    // binary classification problem.
    val binarySummary = trainingSummary.asInstanceOf[BinaryLogisticRegressionSummary]

    // Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    val roc = binarySummary.roc
    roc.show()
    println(s"areaUnderROC: ${binarySummary.areaUnderROC}")

    // Set the model threshold to maximize F-Measure
    val fMeasure = binarySummary.fMeasureByThreshold
    val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
    val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure)
      .select("threshold").head().getDouble(0)
    lrModel.setThreshold(bestThreshold)
  }
}
