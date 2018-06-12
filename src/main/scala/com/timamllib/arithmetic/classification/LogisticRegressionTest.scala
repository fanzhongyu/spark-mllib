package com.timamllib.arithmetic.classification

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{Row, SparkSession}

/*
二项逻辑回归的例子
 */
object LogisticRegressionTest {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("LogisticRegressionTest")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    //Load TrainData
    //Seq类型是有顺序的数据结构
    //training为dataFrame
    val training = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(0.0, 1.1, 0.1)),
      (0.0, Vectors.dense(2.0, 1.0, -1.0)),
      (0.0, Vectors.dense(2.0, 1.3, 1.0)),
      (1.0, Vectors.dense(0.0, 1.2, -0.5))
    )).toDF("label", "features")//转变为数据框
    //查看train Data
    println("training data")
    training.show()
    //[1.0,[0.0,1.1,0.1]]
    //[0.0,[2.0,1.0,-1.0]]

    // 创建logistics regression实例
    val lr = new LogisticRegression()
    // Print out the parameters, documentation, and any default values.
    println("LogisticRegression parameters:\n" + lr.explainParams() + "\n")

    //重新设置模型参数方法
    lr.setMaxIter(10)//最大迭代步数
      .setRegParam(0.01)

    // 根据设定的模型参数与training data拟合训练得到模型
    val model1 = lr.fit(training)
    // Since model1 is a Model (i.e., a Transformer produced by an Estimator),
    // we can view the parameters it used during fit().
    // This prints the parameter (name: value) pairs, where names are unique IDs for this
    println("Model 1 was fit using parameters: " + model1.parent.extractParamMap)

    val testData = spark.createDataFrame(Seq(
      (1.0, Vectors.dense(-1.0, 1.5, 1.3)),
      (0.0, Vectors.dense(3.0, 2.0, -0.1)),
      (1.0, Vectors.dense(0.0, 2.2, -1.5)),
      (1.0, Vectors.dense(2.0, 2.2, -1.0))
    )).toDF("label", "features")

    model1.transform(testData)
      .select("features", "label", "probability", "prediction")//选择数据框的某些列
      .collect()//一般在filter或者足够小的结果的时候，再用collect封装返回一个数组
      .foreach { case Row(features, label: Double, prob, prediction: Double) =>
      println(s"($features, $label) -> prob=$prob, prediction=$prediction")
    }

    // We may alternatively specify parameters using a ParamMap,
    // which supports several methods for specifying parameters.
    val paramMap = ParamMap(lr.maxIter -> 20)
      .put(lr.maxIter, 30) // Specify 1 Param.  This overwrites the original maxIter.
      .put(lr.regParam -> 0.1, lr.threshold -> 0.9) // Specify multiple Params.

    // One can also combine ParamMaps.
    val paramMap2 = ParamMap(lr.probabilityCol -> "myProbability") // 将输出结果列中Change output column name
    val paramMapCombined = paramMap ++ paramMap2//两个参数设置的map合并,两个map的合并.

    // Now learn a new model using the paramMapCombined parameters.
    // paramMapCombined overrides all parameters set earlier via lr.set* methods.
    val model2 = lr.fit(training, paramMapCombined)
    println("Model 2 was fit using parameters: " + model2.parent.extractParamMap)

    // Make predictions on test data using the Transformer.transform() method.
    // LogisticRegression.transform will only use the 'features' column.
    // Note that model2.transform() outputs a 'myProbability' column instead of the usual//输出概率列
    // 'probability' column since we renamed the lr.probabilityCol parameter previously.
    model2.transform(testData)//transform为得到一个新rdd
      .select("features", "label", "myProbability", "prediction")
      .collect()//返回所有selected数据 in an array
      .foreach { case Row(features, label: Double, prob, prediction: Double) =>
      println(s"($features, $label) -> prob=$prob, prediction=$prediction")//自定义输出数据
    }


  }
}
