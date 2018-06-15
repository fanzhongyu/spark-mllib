package com.timamllib.arithmetic.cluster

import breeze.linalg.{DenseVector, sum}
import breeze.numerics.pow
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.{SparkConf, SparkContext}


/*
k均值算法的计算过程非常直观：

      1、从D中随机取k个元素，作为k个簇的各自的中心。

      2、分别计算剩下的元素到k个簇中心的相异度，将这些元素分别划归到相异度最低的簇。

      3、根据聚类结果，重新计算k个簇各自的中心，计算方法是取簇中所有元素各自维度的算术平均数。

      4、将D中全部元素按照新的中心重新聚类。

      5、重复第4步，直到聚类结果不再变化。

      6、将结果输出。
 */
object KmeansCluster {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
    val conf = new SparkConf().setMaster("local[4]").setAppName("Clustering")
    val sc = new SparkContext(conf)
    /*加载电影信息*/
    val file_item=sc.textFile("data/mllib/ml-100k/u.item")
    println(file_item.first())
    /* 1|Toy Story (1995)|01-Jan-1995
     ||http://us.imdb.com/M/title-exact?Toy%20Story%20(1995)|0|0|0|1|1|1|0|0|0|0|0|0|0|0|0|0|0|0|0*/
    /*加载电影类别信息*/
    val file_genre=sc.textFile("data/mllib/ml-100k/u.genre")
    println(file_genre.first())
    /*加载评论人的信息*/
    val file_user=sc.textFile("data/mllib/ml-100k/u.user")
    /*加载评论人的评论信息*/
    val file_data=sc.textFile("data/mllib/ml-100k/u.data")

    /*训练推荐模型*/
    val data_vector=file_data.map(_.split("\t")).map{
      x =>
        Rating(x(0).toInt,x(1).toInt,x(2).toDouble)
    }.cache()
    val aslModel=ALS.train(data_vector,50,10,0.1)
    /*获取用户相似特征*/
    val userFactors=aslModel.userFeatures
    /*用户特征向量化*/
    val userVectors=userFactors.map(x =>Vectors.dense(x._2))
    /*获取商品相似特征*/
    val movieFactors=aslModel.productFeatures
    /*商品相似特征向量化*/
    val movieVectors=movieFactors .map(x => Vectors.dense(x._2))

    val movieMatrix=new RowMatrix(movieVectors)
    val movieMatrix_Summary=movieMatrix.computeColumnSummaryStatistics()
    println(movieMatrix_Summary.mean)//每列的平均值
    println(movieMatrix_Summary.variance)//每列的方差
    val userMatrix=new RowMatrix(userVectors)
    val userMatrix_Summary=userMatrix.computeColumnSummaryStatistics()
    println(userMatrix_Summary.mean)//每列的平均值
    println(userMatrix_Summary.variance)//每列的方差

    /*对用户K-means因子聚类*/
    val userClusterModel=KMeans.train(userVectors,5,100)
    /*使用聚类模型进行预测*/
    val user_predict=userClusterModel.predict(userVectors)
    def computeDistance(v1:DenseVector[Double],v2:DenseVector[Double])=sum(pow(v1-v2,2))
    user_predict.map(x =>(x,1)).reduceByKey(_+_).collect().foreach(println(_))
    /*每个类中的数目
    (4,170)
    (0,230)
    (1,233)
    (2,175)
    (3,135)
    */
    val userInfo=file_user.map(_.split("\\|")).map{
      x => (x(0).toInt,(x(1),x(2),x(3),x(4)))
    }
    /*联合用户信息和特征值*/
    val infoAndFactors=userInfo.join(userFactors)
    val userAssigned=infoAndFactors.map{
      case(userId,((age,sex,title,zip),factors)) =>
        val pred=userClusterModel.predict(Vectors.dense(factors))
        val center=userClusterModel.clusterCenters(pred)
        val dist=computeDistance(DenseVector(factors),DenseVector(center.toArray))
        (userId,age,sex,title,zip,dist,pred)
    }
    val userCluster=userAssigned.groupBy(_._7).collectAsMap()
    /*输出每个类中的20个用户分类情况*/
    for((k,v) <- userCluster.toSeq.sortBy(_._1)){
      println(s"userCluster$k")
      val info=v.toSeq.sortBy(_._6)
      println(info.take(20).map{
        case(userId,age,sex,title,zip,pred,dist) =>
          (userId,age,sex,title,zip)
      }.mkString("\n"))
      println("========================")
    }



    /*对电影K-means因子聚类*/
    val movieClusterModel=KMeans.train(movieVectors,5,100)
    /*KMeans: KMeans converged in 39 iterations.*/
    val movie_predict=movieClusterModel.predict(movieVectors)
    movie_predict.map(x =>(x,1)).reduceByKey(_+_).collect.foreach(println(_))
    /*result
    (4,384)
    (0,340)
    (1,154)
    (2,454)
    (3,350)
     */
    /*查看及分析商品相似度聚类数据*/
    /*提取电影的题材标签*/
    val genresMap=file_genre.filter(!_.isEmpty).map(_.split("\\|"))
      .map(x => (x(1),x(0))).collectAsMap()
    /*为电影数据和题材映射关系创建新的RDD，其中包含电影ID、标题和题材*/
    val titlesAndGenres=file_item.map(_.split("\\|")).map{
      array =>
        val geners=array.slice(5,array.size).zipWithIndex.filter(_._1=="1").map(
          x => genresMap(x._2.toString)
        )
        (array(0).toInt,(array(1),geners))
    }
    val titlesWithFactors=titlesAndGenres.join(movieFactors)
    val movieAssigned=titlesWithFactors.map{
      case(id,((movie,genres),factors)) =>
        val pred=movieClusterModel.predict(Vectors.dense(factors))
        val center=movieClusterModel.clusterCenters(pred)
        val dist=computeDistance(DenseVector(factors),DenseVector(center.toArray))
        (id,movie,genres.mkString(" "),pred,dist)
    }
    val clusterAssigned=movieAssigned.groupBy(_._4).collectAsMap()
    for((k,v)<- clusterAssigned.toSeq.sortBy(_._1)){
      println(s"Cluster$k")
      val dist=v.toSeq.sortBy(_._5)
      println(dist.take(20).map{
        case (id,movie,genres,pred,dist) =>
          (id,movie,genres)
      }.mkString("\n"))
      println("============")
    }
  }
}
