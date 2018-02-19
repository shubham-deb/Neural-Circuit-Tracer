package scala
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.util._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.classification._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.storage.StorageLevel
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel

// args(0) - Input
// args(1) - ValidationInput
// args(2) - ModelOutput

object Model {
  def main(args: Array[String]) = {
    
      val conf = new SparkConf()
                .setAppName("Project")
                
      val sc = new SparkContext(conf)
         
      // Train a RandomForest model using the parameters below.
      val num_of_classes = 2
     //Empty categoricalFeaturesInfo indicates all features are continuous.
      val categoricalFeaturesInfo = Map[Int, Int]() 
      // more trees reduce variance
      val numTrees = 30
      val featureSubsetStrategy = "auto"
      val impurity = "gini"
      val max_depth = 4
      val maxBins = 100 
      val filename = args(0)
      val validation = args(1)
      
      val rdd_data: RDD[(Double,Array[Double])] = sc.textFile(filename).map{ line =>
        val parsed_line = line.split(",")
        (parsed_line(parsed_line.length-1).toDouble ,(parsed_line.slice(0,parsed_line.length-1).map(_.toDouble)))
      }
      
      // get the total number of ones and zeroes in rdd_data
      val ones: RDD[(Double,Array[Double])] = rdd_data.filter { case (key, value) => key == 1.0 }
      val zeroes: RDD[(Double,Array[Double])] = rdd_data.filter { case (key, value) => key == 0.0 }
      
      // Then we randomly split the zeroes
      val Array(trainingZeroes, testZeroes) = zeroes.randomSplit(Array(0.1, 0.9), seed = 11L)
      
      val training: RDD[LabeledPoint] = ones.union(trainingZeroes)
                                                     .map{case (key, value) =>
                                                       LabeledPoint(key ,Vectors.dense(value))}
      val test: RDD[LabeledPoint] = ones.union(zeroes)
                                            .map{case (key, value) =>
                                              LabeledPoint(key ,Vectors.dense(value))} 

      val startTime = System.nanoTime()
      
      // Run training algorithm to build the model
      val model = RandomForest.trainClassifier(training,num_of_classes,categoricalFeaturesInfo,
                                          numTrees,featureSubsetStrategy,impurity,max_depth,maxBins)
      
       training.unpersist(true)
      
      // Compute raw scores on the test set
      val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
      }
      
      val numberOfPredictedOnes = predictionAndLabels.filter(r => (r._1 == 1.0 && r._2 == 1.0)).count()
      
      val numberOfWrongPredictedOnes = predictionAndLabels.filter(r => (r._1 == 1.0 && r._2 == 0.0)).count()
      
      val numberOfPredictedZeroes = predictionAndLabels.filter(r => (r._1 == 0.0 && r._2 == 0.0)).count()
      
      val numberOfWrongZeroes = predictionAndLabels.filter(r => (r._1 == 0.0 && r._2 == 1.0)).count()
      
      val testAcc = (numberOfPredictedOnes + numberOfPredictedZeroes)/
                    (numberOfWrongPredictedOnes+numberOfPredictedOnes+numberOfPredictedZeroes+numberOfWrongZeroes).toDouble
      
      val endTime = System.nanoTime()
      
      // Instantiate metrics object
      val metrics = new BinaryClassificationMetrics(predictionAndLabels)
      
      // ROC Curve
      val roc = metrics.roc
      
      // AUROC
      val auROC = metrics.areaUnderROC
      println("Area under ROC = " + auROC)
      
       // Save and load model
      model.save(sc, args(2))
      val sameModel = RandomForestModel.load(sc, args(2))
      
      // map the test data to RDD of labelled points for validation
      val new_rdd_data: RDD[LabeledPoint] = sc.textFile(validation).map{ line =>
        val parsed_line = line.split(",")
        LabeledPoint(parsed_line(parsed_line.length-1).toDouble ,Vectors.dense(parsed_line.slice(0,parsed_line.length-1).map(_.toDouble)))
      }
      
      // Compute raw scores on the validation set
      val predictionAndLabelsForValidation = new_rdd_data.map { case LabeledPoint(label, features) =>
        val prediction = sameModel.predict(features)
        (prediction, label)
      }
      
      val numberOfValidationPredictedOnes = predictionAndLabelsForValidation.filter(r => (r._1 == 1.0 && r._2 == 1.0)).count()
      
      val numberOfValidationWrongPredictedOnes = predictionAndLabelsForValidation.filter(r => (r._1 == 1.0 && r._2 == 0.0)).count()
      
      val numberOfValidationPredictedZeroes = predictionAndLabelsForValidation.filter(r => (r._1 == 0.0 && r._2 == 0.0)).count()
      
      val numberOfValidationWrongZeroes = predictionAndLabelsForValidation.filter(r => (r._1 == 0.0 && r._2 == 1.0)).count()
      
      val ValidationtestAcc = (numberOfValidationPredictedOnes+numberOfValidationPredictedZeroes)/
                  (numberOfValidationWrongPredictedOnes+numberOfValidationPredictedOnes
                      +numberOfValidationWrongZeroes+numberOfValidationPredictedZeroes).toDouble
      
      val new_metrics = new BinaryClassificationMetrics(predictionAndLabelsForValidation)
 
      // ROC Curve
      val new_roc = new_metrics.roc
      
      // AUROC
      val new_auROC = new_metrics.areaUnderROC
      println("Area under ROC = " + new_auROC)
  }
}
