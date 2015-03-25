import com.typesafe.config.ConfigFactory

/**
 * Created by diego on 11/27/14.
 */

object Params {

  val conf = ConfigFactory.load()

  // input file name
  val inputFile = conf.getString("General.InputFile")

  // number of iterations
  val numberOfIterations = conf.getInt("Algo.NumberOfIterations")

  // step size
  val stepSize = conf.getDouble("Algo.StepSize")

  // regularization parameter
  val regularizationValue = conf.getDouble("Algo.RegularizationValue")

  // algorithm type
  val algoType = conf.getString("Algo.Type")

  // rho for admm
  val rho = conf.getDouble("AlgoADMM.Rho")

  // stopping epsilon
  var stoppingEpsilon = conf.getDouble("Algo.StoppingEspilon")

  override def toString:String = {
    "inputFile " + inputFile + " NumberOfIterations " + numberOfIterations + " StepSize " + stepSize + " RegularizationValue " + regularizationValue + " Type " + algoType + " Rho " + rho
  }
}
