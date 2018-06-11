package com.timamllib.feature.transformers

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/*
VectorAssembler 是将给定的一系列的列合并到单个向量列中的 transformer。
它可以将原始特征和不同特征transformers（转换器）生成的特征合并为单个特征向量，
来训练 ML 模型,如逻辑回归和决策树等机器学习算法。VectorAssembler 可接受以下的输入列类型：
所有数值型、布尔类型、向量类型。输入列的值将按指定顺序依次添加到一个向量中。
 */
object VectorAssembler {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("VectorAssembler")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    val dataset = spark.createDataFrame(
      Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5), 1.0))
    ).toDF("id", "hour", "mobile", "userFeatures", "clicked")

    val assembler = new VectorAssembler()
      .setInputCols(Array("hour", "mobile", "userFeatures"))
      .setOutputCol("features")

    val output = assembler.transform(dataset)
    /*
    [[18.0,1.0,0.0,10.0,0.5],1.0]
     */
    println(output.select("features", "clicked").first())
  }
}
