package com.timamllib.feature.transformers

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.sql.SparkSession

object VectorIndexer {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("Tokenizer")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    val data = spark.read.format("libsvm").load("mllib/sample_libsvm_data.txt")

    val indexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexed")
      .setMaxCategories(10)

    val indexerModel = indexer.fit(data)

    val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
    println(s"Chose ${categoricalFeatures.size} categorical features: " +
      categoricalFeatures.mkString(", "))

    // Create new column "indexed" with categorical values transformed to indices
    val indexedData = indexerModel.transform(data)
    indexedData.show()
  }
}
