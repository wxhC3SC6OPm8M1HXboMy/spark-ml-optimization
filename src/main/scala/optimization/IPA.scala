package optimization

import com.github.fommil.netlib.BLAS

import org.apache.spark.mllib.optimization.Gradient
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector,Vectors,DenseVector}
import org.apache.spark.mllib.rdd.RDDFunctions._

import scala.collection.mutable.ArrayBuffer

import optimization.{Updater=>DistUpdater}

/**
 * Created by diego on 1/28/15.
 */

class IPA(private var gradient: Gradient, private var updater: DistUpdater) extends Distributed(gradient,updater) {

  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val stepFunction = this.stepSizeFunction match {
      case Some(func) => func
      case None => if(regParam > 0) stepSizeFunctionWithReg else stepSizeFunctionNoReg
    }
    val (weights, _) = IPA.runIPA(
      data,
      gradient,
      updater,
      stepSize,
      stoppingEpsilon,
      stepFunction,
      numIterations,
      regParam,
      initialWeights)
    weights
  }
}

/*
 * run distributed ipa
 * return: weights, loss in each iteration
 * loss = sum of losses for each record based on current weight at that local iteration + regularization value of the weight for the next iteration
 */

object IPA extends Logging {
  private val MAX_LOSSES_TO_REPORT:Int = 20

  def runIPA(
                       data: RDD[(Double, Vector)],
                       gradient: Gradient,
                       updater: DistUpdater,
                       stepSize: Double,
                       stoppingEpsilon: Double,
                       stepSizeFunction: (Int) => Double,
                       numIterations: Int,
                       regParam: Double,
                       initialWeights: Vector): (Vector, Array[Double]) = {

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    var weights = Vectors.dense(initialWeights.toArray)

    val numberOfFeatures = weights.size
    val noPartitions = data.partitions.length
    val zeroVector = Vectors.zeros(numberOfFeatures)

    val noRecords = data.count()
    val newRegParam = regParam/noRecords

    var actualIterations = 0

    def runOneIteration(j:Int,stopFlag: Boolean): Unit = {
      if(j < 0 || stopFlag) return
      // broadcast weights
      val bcWeights = data.context.broadcast(weights)
      // for each partition in the rdd
      val oneIterRdd = data.mapPartitions{ case iter =>
        var w = bcWeights.value.copy
        val originalWeight = bcWeights.value
        var iterCount = 1
        var loss = 0.0
        while(iter.hasNext) {
          val (label,features) = iter.next()
          // gradient
          val (newGradient,_) = gradient.compute(features, label, w)
          val (_,newLoss) = gradient.compute(features,label,originalWeight)
          // update current point
          val (w1,_) = updater.compute(w, newGradient, stepSize, stepSizeFunction, iterCount, newRegParam)
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

      // check if the previous vector weights and the new one sumWeight differ by more then stoppingEspilon
      val diffArray = new Array[Double](sumWeight.size)
      BLAS.getInstance().dcopy(sumWeight.size,sumWeight.asInstanceOf[DenseVector].values,1,diffArray,1)
      BLAS.getInstance().daxpy(sumWeight.size,-1.0,weights.asInstanceOf[DenseVector].values,1,diffArray,1)
      val normDiff = BLAS.getInstance().dnrm2(sumWeight.size,diffArray,1)

      weights = sumWeight

      // to compute the regularization value
      val regVal = updater.compute(weights, zeroVector, 0, x => x, 1, regParam)._2

      actualIterations += 1
      stochasticLossHistory.append(totalLoss+regVal)

      bcWeights.destroy()

      val stop = if(normDiff < stoppingEpsilon) true else false

      runOneIteration(j-1,stop)
    }
    runOneIteration(numIterations-1,stopFlag = false)

    val noLossesToReport = math.min(MAX_LOSSES_TO_REPORT,actualIterations)
    logInfo("IPA finished. Last %d losses %s".format(noLossesToReport,stochasticLossHistory.takeRight(noLossesToReport).mkString(", ")))

    (weights,stochasticLossHistory.toArray)
  }
}
