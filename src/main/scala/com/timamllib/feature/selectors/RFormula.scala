package com.timamllib.feature.selectors

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.sql.SparkSession

/**
RFormula选择由R模型公式（R model formula）指定的列。目前，我们支持R运算符的有限子集，
包括‘~’, ‘.’, ‘:’, ‘+’以及‘-‘，基本操作如下：

~分隔目标和对象
+合并对象，“+ 0”表示删除截距
- 删除对象，“ - 1”表示删除截距
:交互（数字乘法或二值化分类值）
. 除了目标外的全部列
假设a和b是double列，我们使用以下简单的例子来说明RFormula的效果：
 y ~ a + b表示模型y ~ w0 + w1 * a +w2 * b其中w0为截距，w1和w2为相关系数。
 y ~a + b + a:b – 1表示模型y ~ w1* a + w2 * b + w3 * a * b，其中w1，w2，w3是相关系数。
RFormula产生一个特征向量列和一个标签的double列或label列。
像R在线性回归中使用公式时，字符型的输入将转换成one-hot编码，数字列将被转换为双精度。
如果label列是类型字符串，则它将首先使用StringIndexer转换为double。 如果DataFrame中不存在label列，
则会从公式中指定的响应变量创建输出标签列。
*/
object RFormula {
  def main(args: Array[String]): Unit = {
    val sparkConf = new SparkConf().setMaster("spark://172.20.61.32:7077").setAppName("RFormula")

    val spark = SparkSession.builder().config(sparkConf).getOrCreate()

    val dataset = spark.createDataFrame(Seq(
      (7, "US", 18, 1.0),
      (8, "CA", 12, 0.0),
      (9, "NZ", 15, 0.0)
    )).toDF("id", "country", "hour", "clicked")

    val formula = new RFormula()
      .setFormula("clicked ~ country + hour")
      .setFeaturesCol("features")
      .setLabelCol("label")

    val output = formula.fit(dataset).transform(dataset)
    output.select("features", "label").show()

  }
}
