package optimization

import scala.math._

import breeze.linalg.{norm => brzNorm, axpy => brzAxpy, Vector => BV}
import org.apache.spark.mllib.linalg.{Vector}

/**
 *  same as the regular spark updaters except that the step size formula is more generic, pluggable
 */

/**
  *  Class used to perform steps (weight update) using Gradient Descent methods.
  * 
  *  For general minimization problems, or for regularized problems of the form
  *          min  L(w) + regParam * R(w),
  *  the compute function performs the actual update step, when given some
  *  (e.g. stochastic) gradient direction for the loss L(w),
  *  and a desired step-size (learning rate).
  * 
  *  The updater is responsible to also perform the update coming from the
  *  regularization term R(w) (if any regularization is used).
  */

abstract class Updater extends Serializable {
  /**
    *  Compute an updated value for weights given the gradient, stepSize, iteration number and
    *  regularization parameter. Also returns the regularization value regParam * R(w)
    *  computed using the *updated* weights.
    * 
    *  @param weightsOld - Column matrix of size dx1 where d is the number of features.
    *  @param gradient - Column matrix of size dx1 where d is the number of features.
    *  @param stepSize - step size across iterations
    *  @param iter - Iteration number
    *  @param regParam - Regularization parameter
    * 
    *  @return A tuple of 2 elements. The first element is a column matrix containing updated weights,
    *          and the second element is the regularization value computed using updated weights.
    */
  def compute(
      weightsOld: Vector,
      gradient: Vector,
      stepSize: Double,
      stepSizeFunction: (Int) => Double,
      iter: Int,
      regParam: Double): (Vector, Double)
}

/**
  *  :: DeveloperApi ::
  *  A simple updater for gradient descent *without* any regularization.
  *  Uses a step-size decreasing with the square root of the number of iterations.
  */

class SimpleUpdater extends Updater {
  override def compute(
      weightsOld: Vector,
      gradient: Vector,
      stepSize: Double,
      stepSizeFunction: (Int) => Double,
      iter: Int,
      regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize * stepSizeFunction(iter)
    val brzWeights: BV[Double] = BreezeHelper.toBreeze(weightsOld).toDenseVector
    // val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    // brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    brzAxpy(-thisIterStepSize, BreezeHelper.toBreeze(gradient), brzWeights)

    (BreezeHelper.fromBreeze(brzWeights), 0)
  }
}

/**
  *  :: DeveloperApi ::
  *  Updater for L1 regularized problems.
  *           R(w) = ||w||_1
  *  Uses a step-size decreasing with the square root of the number of iterations.
  *  Instead of subgradient of the regularizer, the proximal operator for the
  *  L1 regularization is applied after the gradient step. This is known to
  *  result in better sparsity of the intermediate solution.
  * 
  *  The corresponding proximal operator for the L1 norm is the soft-thresholding
  *  function. That is, each weight component is shrunk towards 0 by shrinkageVal.
  * 
  *  If w >  shrinkageVal, set weight component to w-shrinkageVal.
  *  If w < -shrinkageVal, set weight component to w+shrinkageVal.
  *  If -shrinkageVal < w < shrinkageVal, set weight component to 0.
  * 
  *  Equivalently, set weight component to signum(w) * max(0.0, abs(w) - shrinkageVal)
  */

class L1Updater extends Updater {
  override def compute(
      weightsOld: Vector,
      gradient: Vector,
      stepSize: Double,
      stepSizeFunction: (Int) => Double,
      iter: Int,
      regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize * stepSizeFunction(iter)
    // Take gradient step
    //val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    //brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    val brzWeights: BV[Double] = BreezeHelper.toBreeze(weightsOld).toDenseVector
    brzAxpy(-thisIterStepSize, BreezeHelper.toBreeze(gradient), brzWeights)
    // Apply proximal operator (soft thresholding)
    val shrinkageVal = regParam * thisIterStepSize
    var i = 0
    while (i < brzWeights.length) {
      val wi = brzWeights(i)
      brzWeights(i) = signum(wi) * max(0.0, abs(wi) - shrinkageVal)
      i += 1
    }

    (BreezeHelper.fromBreeze(brzWeights), brzNorm(brzWeights, 1.0) * regParam)
  }
}

/**
  *  :: DeveloperApi ::
  *  Updater for L2 regularized problems.
  *           R(w) = 1/2 pow(||w||,2)
  *  Uses a step-size decreasing with the square root of the number of iterations.
  */

class SquaredL2Updater extends Updater {
  override def compute(
      weightsOld: Vector,
      gradient: Vector,
      stepSize: Double,
      stepSizeFunction: (Int) => Double,
      iter: Int,
      regParam: Double): (Vector, Double) = {
    // add up both updates from the gradient of the loss (= step) as well as
    // the gradient of the regularizer (= regParam * weightsOld)
    // w' = w - thisIterStepSize * (gradient + regParam * w)
    // w' = (1 - thisIterStepSize * regParam) * w - thisIterStepSize * gradient
    val thisIterStepSize = stepSize * stepSizeFunction(iter)
    // val brzWeights: BV[Double] = weightsOld.toBreeze.toDenseVector
    val brzWeights: BV[Double] = BreezeHelper.toBreeze(weightsOld).toDenseVector
    brzWeights :*= (1.0 - thisIterStepSize * regParam)
    //brzAxpy(-thisIterStepSize, gradient.toBreeze, brzWeights)
    brzAxpy(-thisIterStepSize, BreezeHelper.toBreeze(gradient), brzWeights)
    val norm = brzNorm(brzWeights, 2.0)

    (BreezeHelper.fromBreeze(brzWeights), 0.5 * regParam * norm * norm)
  }
}
