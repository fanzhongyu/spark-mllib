package com.timamllib.feature.transformers

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.SQLTransformer
import org.apache.spark.sql.SparkSession

/**
  * SQLTransformer（SQL转换器）实现由 SQL 语句定义的转换 。目前我们只支持SQL语法如
*"SELECT ... FROM __THIS__ ..." ，其中 "__THIS__" 代表输入数据集的基础表。
*选择语句指定输出中展示的字段、元素和表达式，支持Spark SQL 中的所有选择语句。
*用户还可以使用 Spark SQL 内置函数和U DFs（自定义函数）来对这些选定的列进行操作。
*SQLTransformer 支持如下语句：
 **
 SELECT a, a + b AS a_b FROM __THIS__
    *SELECT a, SQRT(b) AS b_sqrt FROM __THIS__ where a > 5
    *SELECT a, b, SUM(c) AS c_sum FROM __THIS__ GROUP BY a, b
  */
object SQLTransformer {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("SQLTransformer")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    val df = spark.createDataFrame(
      Seq((0, 1.0, 3.0), (2, 2.0, 5.0))).toDF("id", "v1", "v2")

    val sqlTrans = new SQLTransformer().setStatement(
      "SELECT *, (v1 + v2) AS v3, (v1 * v2) AS v4 FROM __THIS__")

    sqlTrans.transform(df).show()
  }
}
