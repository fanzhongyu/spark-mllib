package com.timamllib.feature.transformers

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * MinMaxScaler转换Vector行的数据集，将每个要素的重新映射到特定范围（通常为[0，1]）。它需要参数：
  **
  *min：默认为0.0，转换后的下限，由所有功能共享。
  **
  *max：默认为1.0，转换后的上限，由所有功能共享。
  **
  *MinMaxScaler计算数据集的统计信息，并生成MinMaxScalerModel。然后，模型可以单独转换每个要素，使其在给定的范围内。
  */
object MinMaxScaler {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("StringIndexer")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    val dataFrame = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.1, -1.0)),
      (1, Vectors.dense(2.0, 1.1, 1.0)),
      (2, Vectors.dense(3.0, 10.1, 3.0))
    )).toDF("id", "features")

    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")

    // Compute summary statistics and generate MinMaxScalerModel
    val scalerModel = scaler.fit(dataFrame)

    // rescale each feature to range [min, max].
    val scaledData = scalerModel.transform(dataFrame)
    println(s"Features scaled to range: [${scaler.getMin}, ${scaler.getMax}]")
    scaledData.select("features", "scaledFeatures").show()
  }
}
