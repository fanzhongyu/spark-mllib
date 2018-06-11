package com.timamllib.feature.transformers

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.MaxAbsScaler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
MaxAbsScaler转换Vector行的数据集，通过划分每个要素中的最大绝对值，将每个要素的重新映射到范围[-1,1]。
它不会使数据移动/居中，因此不会破坏任何稀疏性。

MaxAbsScaler计算数据集的统计信息，并生成MaxAbsScalerModel。
然后，模型可以将每个要素单独转换为范围[-1,1]。
  */
object MaxAbsScaler {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("MaxAbsScaler")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    val dataFrame = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.1, -8.0)),
      (1, Vectors.dense(2.0, 1.0, -4.0)),
      (2, Vectors.dense(4.0, 10.0, 8.0))
    )).toDF("id", "features")

    val scaler = new MaxAbsScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")

    // Compute summary statistics and generate MaxAbsScalerModel
    val scalerModel = scaler.fit(dataFrame)

    // rescale each feature to range [-1, 1]
    val scaledData = scalerModel.transform(dataFrame)
    scaledData.select("features", "scaledFeatures").show()
  }
}
