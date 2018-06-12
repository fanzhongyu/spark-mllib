package com.timamllib.arithmetic.classification

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

object MultinomialLogisticRegression {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("local").setAppName("MultinomialLogisticRegression")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    // Load training data
    val training = spark
      .read
      .format("libsvm")
      .load("data/mllib/sample_multiclass_classification_data.txt")

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(training)

    // Print the coefficients and intercept for multinomial logistic regression
    println(s"Coefficients: \n${lrModel.coefficientMatrix}")
    println(s"Intercepts: ${lrModel.interceptVector}")
  }
}
