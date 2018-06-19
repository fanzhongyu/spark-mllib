package com.timamllib.arithmetic.Regression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object LinearRegressionTest {
  def labeledFile(originFile: String, sc: SparkContext): RDD[LabeledPoint] = {
    val file_load = sc.textFile(originFile)
    val file_split = file_load.map(_.split(","))

    /*构建映射类函数的方法:mapping*/
    def mapping(rdd: RDD[Array[String]], index: Int) =
      rdd.map(x => x(index)).distinct.zipWithIndex().collect.toMap

    /*存储每列映射方法mapping的maps集合*/
    var maps: Map[Int, Map[String, Long]] = Map()
    /* 生成maps*/
    for (i <- 2 until 10)
      maps += (i -> mapping(file_split, i))
    /*max_size表示每列的特征之和*/
    val max_size = maps.map(x => x._2.size).sum
    val file_label = file_split.map {
      x =>
        var num: Int = 0
        var size: Int = 0
        /*构建长度为max_size+4的特征数组,初始值全为0*/
        val arrayOfDim = Array.ofDim[Double](max_size + 4)
        for (j <- 2 until 10) {
          num = maps(j)(x(j)).toInt
          if (j == 2) size = 0 else size += maps(j - 1).size
          /*为特征赋值*/
          arrayOfDim(size + num) = 1.0
        }
        /*添加后面4列归一化的特征*/
        for (j <- 10 until 14)
          arrayOfDim(max_size + (j - 10)) = x(j).toDouble
        /*生成LabeledPoint类型*/
        LabeledPoint(x(14).toDouble + x(15).toDouble, Vectors.dense(arrayOfDim))
    }
    file_label
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("regression").setMaster("local[4]")
    val sc = new SparkContext(conf)
    //文件名
    val file_bike = "hour_nohead.csv"
    //调用二元向量化方法
    val labeled_file = labeledFile(file_bike, sc)
    /*/*对目标值取对数*/
    val labeled_file1=labeled_file.map(point => LabeledPoint(math.log(point.label),point.features))
    */
    //构建线性回归模型，注该方法在：spark2.1.0已经抛弃了。。。。
    val model_liner = LinearRegressionWithSGD.train(labeled_file, 10, 0.1)
    //val categoricalFeaturesInfo = Map[Int,Int]()
    //val model_DT=DecisionTree.trainRegressor(labeled_file,categoricalFeaturesInfo,"variance",5,32)
    val predict_vs_train = labeled_file.map {
      point => (model_liner.predict(point.features), point.label)
      //对目标取对数后的，预测方法
      /* point => (math.exp(model_liner.predict(point.features)),math.exp(point.label))*/
    }
    predict_vs_train.take(5).foreach(println(_))

    /*
 (135.94648455498356,16.0)
 (134.38058174607252,40.0)
 (134.1840793861374,32.0)
 (133.88699144084515,13.0)
 (133.77899037657548,1.0)
    */
  }
}
