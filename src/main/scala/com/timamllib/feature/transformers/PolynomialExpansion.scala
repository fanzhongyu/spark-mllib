package com.timamllib.feature.transformers

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

//Polynomial expansion （多项式展开）
// 是将特征扩展为多项式空间的过程，多项式空间由原始维度的n度组合组成。
object PolynomialExpansion {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("PolynomialExpansion")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    val data = Array(
      Vectors.dense(2.0, 1.0),
      Vectors.dense(0.0, 0.0),
      Vectors.dense(3.0, -1.0)
    )
    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val polyExpansion = new PolynomialExpansion()
      .setInputCol("features")
      .setOutputCol("polyFeatures")
      .setDegree(3)

    val polyDF = polyExpansion.transform(df)
    polyDF.show(false)
  }
}
