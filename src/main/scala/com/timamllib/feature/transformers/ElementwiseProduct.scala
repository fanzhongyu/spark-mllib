package com.timamllib.feature.transformers

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.ElementwiseProduct
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * ElementwiseProduct 将每个输入向量中乘以一个 weight（权重）向量，
  * 使用元素相乘的方法.换句话来说,就是通过scalar multiplier （标量乘法）对数据集中的每一列进行缩放。
  * 这表示输入向量 v 和转换向量 w 通过 Hadamard product（Hadamard积） 产生一个结果向量.
  */
object ElementwiseProduct {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("ElementwiseProduct")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    // Create some vector data; also works for sparse vectors
    val dataFrame = spark.createDataFrame(Seq(
      ("a", Vectors.dense(1.0, 2.0, 3.0)),
      ("b", Vectors.dense(4.0, 5.0, 6.0)))).toDF("id", "vector")

    val transformingVector = Vectors.dense(0.0, 1.0, 2.0)
    val transformer = new ElementwiseProduct()
      .setScalingVec(transformingVector)
      .setInputCol("vector")
      .setOutputCol("transformedVector")

    /*
     +---+-------------+-----------------+
     | id|       vector|transformedVector|
     +---+-------------+-----------------+
     |  a|[1.0,2.0,3.0]|    [0.0,2.0,6.0]|
     |  b|[4.0,5.0,6.0]|   [0.0,5.0,12.0]|
     +---+-------------+-----------------+
     */
    // Batch transform the vectors to create new column:
    transformer.transform(dataFrame).show()
  }
}
