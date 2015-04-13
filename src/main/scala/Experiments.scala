import java.io._

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.Optimizer

import org.apache.log4j.Logger

import classification._
import optimization.Distributed

import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS,LogisticRegressionWithSGD,SVMWithSGD}


/**
 * Created by diego on 1/31/15.
 */

object Experiments {

  val usage = """
                Usage: Experiments --master opt1 (optional) --conf filename (required)
              """

  def main(args: Array[String]): Unit = {

    if(args.length == 0) {
      println(usage)
      sys.exit(0)
    }

    val arglist = args.toList
    type OptionMap = Map[Symbol, Any]
    def nextOption(map : OptionMap, list: List[String]) : OptionMap = {
      list match {
        case Nil => map
        case "--conf" :: value :: tail =>
          nextOption(map ++ Map('conf -> value.toString), tail)
        case "--master" :: value :: tail =>
          nextOption(map ++ Map('master -> value.toString), tail)
        case option :: tail => println("Unknown option " + option)
          sys.exit(0)
      }
    }
    val options = nextOption(Map(),arglist)

    if(options.contains('conf)) System.setProperty("config.file", options('conf).toString) else sys.exit(0)

    val log = Logger.getLogger("mlAlgorithms")
    log.info(Params.toString)

    val conf = new SparkConf().setAppName("mlOptimization")
    if(options.contains('master)) conf.setMaster(options('master).toString)
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
    val algo = (Params.algoType,Params.modelType) match {
      case ("IPA","LogisticRegression") =>
        new LogisticRegressionWithIPA()
      case ("ADMM","LogisticRegression") =>
        val a = new LogisticRegressionWithADMM()
        a.optimizer.setRho(Params.rho)
        a
      case ("PH","LogisticRegression") =>
        val a = new LogisticRegressionWithPH()
        a.optimizer.setRho(Params.rho)
        a
      case ("PHDistributeRegularizationTerm","LogisticRegression") =>
        val a = new LogisticRegressionWithPHDistributeRegularizationTerm()
        a.optimizer.setRho(Params.rho)
        a
      case ("SGD","LogisticRegression") =>
        val a = new LogisticRegressionWithSGD()
        a.optimizer.setNumIterations(Params.numberOfIterations).setRegParam(Params.regularizationValue)
        a
      case ("BFGS","LogisticRegression") =>
        val a = new LogisticRegressionWithLBFGS()
        a.optimizer.setNumIterations(Params.numberOfIterations).setRegParam(Params.regularizationValue)
        a
      case ("IPA","SVM") =>
        new SVMWithIPA()
      case ("ADMM","SVM") =>
        val a = new SVMWithADMM()
        a.optimizer.setRho(Params.rho)
        a
      case ("PH","SVM") =>
        val a = new SVMWithPH()
        a.optimizer.setRho(Params.rho)
        a
      case ("PHDistributeRegularizationTerm","SVM") =>
        val a = new SVMWithPHDistributeRegularizationTerm()
        a.optimizer.setRho(Params.rho)
        a
      case ("SGD","SVM") =>
        val a = new SVMWithSGD()
        a.optimizer.setNumIterations(Params.numberOfIterations).setRegParam(Params.regularizationValue)
        a
      case ("BFGS","SVM") =>
        val a = new SVMWithBFGS()
        a.optimizer.setNumIterations(Params.numberOfIterations).setRegParam(Params.regularizationValue)
        a
    }

    if(Params.algoType != "SGD" && Params.algoType != "BFGS")
      algo.optimizer.asInstanceOf[Distributed].setNumIterations(Params.numberOfIterations).setRegParam(Params.regularizationValue).setStepSize(Params.stepSize).setStoppingEpsilon(Params.stoppingEpsilon)
    val model = algo.run(inputData)

    log.info("Solved")

    val pw = new PrintWriter(new File("weights.txt"))
    pw.write(model.weights.toArray.mkString(" "))
    pw.close()

    inputData.unpersist()
  }

}
