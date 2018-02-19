package scala
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel

// args(0) - Input
// args(1) - Model Input
// args(2) - Output

object ScalaProject {
  
  def main(args: Array[String]) = {
        //Start the Spark context
    val conf = new SparkConf()
                .setAppName("Project")

    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val filename = args(0)
                   
   // retreive the model
   val sameModel = RandomForestModel.load(sc, args(1))
   
   // get the RDD of labelled points
    val new_rdd_data= sc.textFile(filename).map{ line =>
        val parsed_line = line.split(",")
        LabeledPoint(0 ,Vectors.dense(parsed_line.slice(0,parsed_line.length-1).map(_.toDouble)))
      }
      
    
      val startTime = System.nanoTime()
      
      // Compute raw scores on the validation set
      val predictionAndLabelsForValidation = new_rdd_data.map { case LabeledPoint(label, features) =>
        val prediction = sameModel.predict(features)
        (prediction.toInt)
      }      
      
      val endTime = System.nanoTime()
      
      predictionAndLabelsForValidation.repartition(1).saveAsTextFile(args(2));
     sc.stop()
  }
}
