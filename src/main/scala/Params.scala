import com.typesafe.config.ConfigFactory

/**
 * Created by diego on 11/27/14.
 */

object Params {

  val conf = ConfigFactory.load()

  // input file name
  val inputFile = conf.getString("General.InputFile")

  // input file name
  val modelType = conf.getString("General.Model")

  // number of iterations
  val numberOfIterations = conf.getInt("Algo.NumberOfIterations")

  // step size
  val stepSize = conf.getDouble("Algo.StepSize")

  // regularization parameter
  val regularizationValue = conf.getDouble("Algo.RegularizationValue")

  // algorithm type
  val algoType = conf.getString("Algo.Type")

  // rho for admm
  val rho = conf.getDouble("AlgoADMMType.Rho")

  // stopping epsilon
  var stoppingEpsilon = conf.getDouble("Algo.StoppingEpsilon")

  override def toString:String = {
    "Input File " + inputFile + " Model Type " + modelType + "  Number Of Iterations " + numberOfIterations + " Step Size " + stepSize + " Regularization Value " + regularizationValue +
      " Algorithm Type " + algoType + " Rho " + rho
  }
}
