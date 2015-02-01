package params

import com.typesafe.config.{ConfigObject, ConfigValue, ConfigFactory}
import scala.collection.JavaConverters._
import java.util.Map.Entry

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

  override def toString:String = {
    "inputFile " + inputFile + " NumberOfIterations " + numberOfIterations + " StepSize " + stepSize + " RegularizationValue " + regularizationValue
  }
}
