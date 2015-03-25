import java.io._

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

import org.apache.log4j.Logger

import classification.{LogisticRegressionWithADMM,LogisticRegressionWithIPA}
import optimization.Distributed

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
      val vt = new Array[Double](v.length-1)
      Array.copy(v,1,vt,0,vt.length)
      val features = Vectors.dense(vt)
      if(v(0) == -1.0) v(0) = 0.0
      LabeledPoint(v(0),features)
    }.persist()
    log.info("Data read. Number of partitions " + inputData.partitions.length)

    /*
     * classification model calibration
     */

    log.info("Solving model")
    val algo = Params.algoType match {
      case "IPA" =>
        new LogisticRegressionWithIPA()
      case "ADMM" =>
        val a = new LogisticRegressionWithADMM()
        a.optimizer.setRho(Params.rho)
        a
    }
    log.info("Solved")

    algo.optimizer.asInstanceOf[Distributed].setNumIterations(Params.numberOfIterations).setRegParam(Params.regularizationValue).setStepSize(Params.stepSize).setStoppingEpsilon(Params.stoppingEpsilon)
    val model = algo.run(inputData)

    val pw = new PrintWriter(new File("weights.txt"))
    pw.write(model.weights.toArray.mkString(" "))
    pw.close()

    inputData.unpersist()
  }

}
