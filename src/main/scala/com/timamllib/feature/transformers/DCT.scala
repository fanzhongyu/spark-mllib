package com.timamllib.feature.transformers

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.DCT
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * Discrete Cosine Transform（离散余弦变换）
  * 是将时域的N维实数序列转换成频域的N维实数序列的过程（有点类似离散傅里叶变换）。
  * （ML中的）DCT类提供了离散余弦变换DCT-II的功能，将离散余弦变换后结果乘以1/根号2
  * 得到一个与时域矩阵长度一致的矩阵。没有偏移被应用于变换的序列
  * （例如，变换的序列的第0个元素是第0个DCT系数，而不是第N / 2个），
  * 即输入序列与输出之间是一一对应的。
  */
object DCT {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("DCT")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    val data = Seq(
      Vectors.dense(0.0, 1.0, -2.0, 3.0),
      Vectors.dense(-1.0, 2.0, 4.0, -7.0),
      Vectors.dense(14.0, -2.0, -5.0, 1.0))

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val dct = new DCT()
      .setInputCol("features")
      .setOutputCol("featuresDCT")
      .setInverse(false)

    val dctDf = dct.transform(df)
    dctDf.select("featuresDCT").show(false)
  }
}
