import java.io._

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

import org.apache.log4j.Logger

import classification.{LogisticRegressionWithADMM,LogisticRegressionWithIPA}

/**
 * Created by diego on 1/31/15.
 */

object Experiments {

  def main(args: Array[String]): Unit = {

    if(args.length > 0) System.setProperty("config.file", args(0))

    val log = Logger.getLogger("mlAlgorithms")
    log.info(Params.toString)

    val conf = new SparkConf().setAppName("mlOptimization")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)

    /*
     * create the rdd of input data
     */

    log.info("Reading data")
    val inputData = sc.textFile(Params.inputFile).map{ t =>
      val v = t.split(" ").map{ _.toDouble }
      val features = Vectors.dense(v)
      if(v(0) == -1.0) v(0) = 0.0
      LabeledPoint(v(0),features)
    }.persist()
    log.info("Data read. Number of partitions " + inputData.partitions.length)

    /*
     * classification model calibration
     */

    log.info("Solving model")
    val model = Params.algoType match {
      case "IPA" => {
        val algo = new LogisticRegressionWithIPA()
        algo.optimizer.setNumIterations(Params.numberOfIterations).setRegParam(Params.regularizationValue).setStepSize(Params.stepSize)
        algo.run(inputData)

      }
      case "ADMM" => {
        val algo = new LogisticRegressionWithADMM()
        algo.optimizer.setNumIterations(Params.numberOfIterations).setRegParam(Params.regularizationValue).setStepSize(Params.stepSize)
          .setRho(Params.rho)
        algo.run(inputData)
      }
    }
    log.info("Solved")

    val pw = new PrintWriter(new File("weights.txt"))
    pw.write(model.weights.toArray.mkString(" "))
    pw.close()

    inputData.unpersist()
  }

}
