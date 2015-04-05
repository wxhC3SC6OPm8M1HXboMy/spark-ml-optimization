package optimization

import org.apache.spark.mllib.optimization.{Optimizer,Gradient}
import org.apache.spark.Logging

import optimization.{Updater=>DistUpdater}

/**
 * Created by diego on 3/25/15.
 *
 * This is the base for distributed algorithms. It basically holds parameters
 */

abstract class Distributed(private var gradient: Gradient, private var updater: DistUpdater) extends Optimizer with Logging {
  protected var numIterations: Int = 2
  protected var regParam: Double = 1.0
  protected var stepSize: Double = 1.0
  protected var stepSizeFunction: Option[(Int) => Double] = None
  protected var stoppingEpsilon: Double = 1.0E-6

  /*
   *  Set the step size function
   *  default: 1 / (1 + regParam * sqrt(t)) if regParam > 0
   *  if regParam = 0, then 1/sqrt(t)
   */

  protected val stepSizeFunctionWithReg = (iterCount:Int) => {
    1.0 / (1.0 + math.sqrt(iterCount))
  }
  protected val stepSizeFunctionNoReg = (iterCount:Int) => {
    1.0 / math.sqrt(iterCount)
  }

  def setStepSizeFunction(func: (Int) => Double): this.type = {
    this.stepSizeFunction = Some(func)
    this
  }

  /*
   * Set the stopping criterion: if consecutive two weights do not change in norm by more than this value, then stop
   */

  def setStoppingEpsilon(value: Double): this.type = {
    this.stoppingEpsilon = value
    this
  }

  /*
   * Set the number of distributed iterations; default = 10
   */

  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  /*
   * Set the regularization parameter. Default 0.0.
   */

  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /*
   * Set the initial step size of SGD for the first step. Default 1.0.
   * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
   */

  def setStepSize(step: Double): this.type = {
    this.stepSize = step
    this
  }

  /*
   * Set the gradient function (of the loss function of one single data example)
   */

  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }

  /*
   * Set the updater function to actually perform a gradient step in a given direction.
   * The updater is responsible to perform the update from the regularization term as well,
   * and therefore determines what kind or regularization is used, if any.
   */

  def setUpdater(updater: DistUpdater): this.type = {
    this.updater = updater
    this
  }
}
