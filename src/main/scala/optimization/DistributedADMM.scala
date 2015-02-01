package optimization

import com.github.fommil.netlib.BLAS

import org.apache.spark.mllib.optimization.{Optimizer,Gradient,Updater}
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector,Vectors,DenseVector}
import org.apache.spark.mllib.rdd.RDDFunctions._

import scala.collection.mutable.ArrayBuffer

/**
 * Created by diego on 1/28/15.
 */

class DistributedADMM(private var gradient: Gradient, private var updater: Updater) extends Optimizer with Logging {
  private var numIterations: Int = 10
  private var regParam: Double = 0.1
  private var stepSize: Double = 1.0

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

  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = DistributedADMM.runDistributedADMM(
      data,
      gradient,
      updater,
      stepSize: Double,
      numIterations,
      regParam,
      initialWeights)
    weights
  }
}

/*
 * run distributed admm
 * return: weights, loss in each iteration
 * loss = sum of losses for each record based on current weight at that local iteration + regularization value of the weight for the next iteration
 */

object DistributedADMM extends Logging {
  def runDistributedADMM(
                       data: RDD[(Double, Vector)],
                       gradient: Gradient,
                       updater: Updater,
                       stepSize: Double,
                       numIterations: Int,
                       regParam: Double,
                       initialWeights: Vector): (Vector, Array[Double]) = {

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    var weights = Vectors.dense(initialWeights.toArray)

    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("DistributedADMM returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    val numberOfFeatures = weights.size
    val noPartitions = data.partitions.size
    val zeroVector = Vectors.dense(new Array[Double](numberOfFeatures))

    def runOneIteration(j:Int): Unit = {
      if(j < 0) return
      // broadcast weights
      val bcWeights = data.context.broadcast(weights)
      // for each partition in the rdd
      val oneIterRdd = data.mapPartitions{ case iter =>
        var w = bcWeights.value
        var iterCount = 1
        var loss = 0.0
        while(iter.hasNext) {
          val (label,features) = iter.next()
          // gradient
          val (newGradient,newLoss) = gradient.compute(features, label, w)
          // update current point
          val (w1,_) = updater.compute(w, newGradient, stepSize, iterCount, regParam)
          loss += newLoss
          w = w1
          iterCount += 1
        }
        List((w,loss)).iterator
      }

      val (sumWeight,totalLoss) = oneIterRdd.treeAggregate((Vectors.zeros(numberOfFeatures), 0.0))(
        seqOp = (c, v) => {
          // c: (weight,loss), v: (one weight vector, one loss)
          // should be this if mllib.BLAS accessible: org.apache.spark.mllib.linalg.BLAS.axpy(1.0,v._1,c._1)
          BLAS.getInstance().daxpy(numberOfFeatures,1.0,v._1.asInstanceOf[DenseVector].values,1,c._1.asInstanceOf[DenseVector].values,1)
          (c._1,v._2+c._2)
      },
      combOp = (c1, c2) => {
        // c: (weight,loss)
        // should be this if mllib.BLAS accessible: org.apache.spark.mllib.linalg.BLAS.axpy(1.0,c2._1,c1._1)
        BLAS.getInstance().daxpy(numberOfFeatures,1.0,c2._1.asInstanceOf[DenseVector].values,1,c1._1.asInstanceOf[DenseVector].values,1)
        (c1._1, c1._2 + c2._2)
      })

      BLAS.getInstance().dscal(numberOfFeatures,1.0/noPartitions,sumWeight.asInstanceOf[DenseVector].values,1)
      weights = sumWeight.asInstanceOf[Vector]

      // to compute the regularization value
      var regVal = updater.compute(weights, zeroVector, 0, 1, regParam)._2
      stochasticLossHistory.append(totalLoss+regParam*regVal)

      runOneIteration(j-1)
    }
    runOneIteration(numIterations)

    logInfo("Distributed ADMM finished. Last 10 losses %s".format(stochasticLossHistory.takeRight(10).mkString(", ")))

    (weights,stochasticLossHistory.toArray)
  }
}