package com.timamllib.feature.transformers

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.sql.SparkSession

/**
  * splits（分割）：这是个将连续的特征转换为 buckets（区间）的参数. n+1次分割时，
  * 将产生n个 buckets（区间）。
  * 一个bucket（区间）通过范围 [x,y) 中 x , y 来定义除了最后一个 bucket 包含 y 值。Splits（分割）
  * 应该是严格递增的。-inf, inf 之间的值必须明确提供来覆盖所有的 Double 值;另外,Double 值超出
  * splits（分割）指定的值将认为是错误的. 两个splits （拆分）的例子为 Array(Double.NegativeInfinity,
  * 0.0, 1.0, Double.PositiveInfinity)以及Array(0.0, 1.0, 2.0)。
  */
object Bucketizer {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("Bucketizer")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    val splits = Array(Double.NegativeInfinity, -0.5, 0.0, 0.5, Double.PositiveInfinity)

    val data = Array(-0.7,-0.5, -0.3, 0.0, 0.2)
    val dataFrame = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val bucketizer = new Bucketizer()
      .setInputCol("features")
      .setOutputCol("bucketedFeatures")
      .setSplits(splits)

    // Transform original data into its bucket index.
    val bucketedData = bucketizer.transform(dataFrame)
    bucketedData.show()
  }
}
