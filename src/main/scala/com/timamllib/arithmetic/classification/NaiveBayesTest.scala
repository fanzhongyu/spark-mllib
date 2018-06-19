package com.timamllib.arithmetic.classification

import org.apache.spark.mllib.classification.{NaiveBayes,NaiveBayesModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkContext,SparkConf}


/*
朴素贝叶斯的例子
是(1)否(0)| 晴天(0)阴天(1)下雨(2)|热(0)舒适(1)冷(2)|不适(0)适合(1)|低(0)高(1)

由于MLlib对数据的格式有严格的要求
主要是classification.{NaiveBayes，NaiveBayesModel}的要求data format：
类别,特征1 特征2 特征3.....
 */
object NaiveBayesTest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local").setAppName("NaiveBayes")
    val sc = new SparkContext(conf)
    val path = "data/mllib/sample_football_weather.txt"
    val data = sc.textFile(path)
    val parsedData = data.map {
      line =>
        val parts = line.split(',')
        LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    }
    //样本划分train和test数据样本60%用于train
    val splits = parsedData.randomSplit(Array(0.6,0.4),seed = 11L)
    val training =splits(0)
    val test =splits(1)
    println("训练数据是"+training)
    println("测试数据是"+test)
    //获得训练模型,第一个参数为数据，第二个参数为平滑参数，默认为1，可改变
    val model =NaiveBayes.train(training,lambda = 1.0) //对测试样本进行测试 //对模型进行准确度分析
    val predictionAndLabel= test.map(p => (model.predict(p.features),p.label))
    val accuracy =1.0 *predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
    //打印一个预测值 println("NaiveBayes精度----->" + accuracy) //我们这里特地打印一个预测值：假如一天是 晴天(0)凉(2)高(0)高(1) 踢球与否
    println("假如一天是 晴天(0)凉(2)高(0)高(1) 踢球与否:" + model.predict(Vectors.dense(0.0,2.0,0.0,1.0))) //保存model
//    val ModelPath = "data/NaiveBayes_model.obj"
//    model.save(sc,ModelPath) //val testmodel = NaiveBayesModel.load(sc,ModelPath)

  }
}
