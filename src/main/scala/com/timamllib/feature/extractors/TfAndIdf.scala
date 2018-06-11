package com.timamllib.feature.extractors

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

object TfAndIdf {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("TF-IDF")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "Hi I heard about Spark"),
      (0.0, "I wish Java could use case classes"),
      (1.0, "Logistic regression models are neat")
    )).toDF("label", "sentence")

    sentenceData.show()

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    /**
    +-----+--------------------+--------------------+
    |label|            sentence|               words|
    +-----+--------------------+--------------------+
    |  0.0|Hi I heard about ...|[hi, i, heard, ab...|
    |  0.0|I wish Java could...|[i, wish, java, c...|
    |  1.0|Logistic regressi...|[logistic, regres...|
    +-----+--------------------+--------------------+
    */
    val wordsData = tokenizer.transform(sentenceData)

    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

    //使用HashingTF将该句子哈希成特征向量
    /**
    +-----+--------------------+--------------------+--------------------+
    |label|            sentence|               words|         rawFeatures|
    +-----+--------------------+--------------------+--------------------+
    |  0.0|Hi I heard about ...|[hi, i, heard, ab...|(20,[0,5,9,17],[1...|
    |  0.0|I wish Java could...|[i, wish, java, c...|(20,[2,7,9,13,15]...|
    |  1.0|Logistic regressi...|[logistic, regres...|(20,[4,6,13,15,18...|
    +-----+--------------------+--------------------+--------------------+
    */
    val featurizedData = hashingTF.transform(wordsData)
    // alternatively, CountVectorizer can also be used to get term frequency vectors

    //我们使用IDF来重新缩放特征向量；这通常会在使用文本作为功能时提高性能。然后，我们的特征向量可以被传递给学习算法。
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    /**
    +-----+--------------------+
    |label|            features|
    +-----+--------------------+
    |  0.0|(20,[0,5,9,17],[0...|
    |  0.0|(20,[2,7,9,13,15]...|
    |  1.0|(20,[4,6,13,15,18...|
    +-----+--------------------+
    */
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("label", "features").show()
  }
}
