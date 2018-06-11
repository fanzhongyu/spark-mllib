package com.timamllib.feature.transformers

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.sql.SparkSession

object StandardScaler {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("StandardScaler")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    // Compute summary statistics by fitting the StandardScaler.
    val scalerModel = scaler.fit(dataFrame)

    // Normalize each feature to have unit standard deviation.
    val scaledData = scalerModel.transform(dataFrame)
    scaledData.show()
  }
}
