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
 *
 * Classical ADMM algorithm
 *
 * Solves by using ADMM: sum_{partition i} sum_{records r in i) loss(r,w)+ regParam * reg(w)
 *
 * Each partition solves: sum_{records r in i) [loss(r,w)+ rho/2 * pow(w-regPenalty+penalty_i),2)/n_i]
 * n_i = the number of records in partition i
 * This problem is solved by single pass updating the gradient
 */

class ADMM(private var gradient: Gradient, private var updater: DistUpdater, private var regularizationOptimizer: RegularizationOptimizationDistributed) extends Distributed(gradient,updater) {
  private var rho: Double = 0.01

  /*
   * set rho; default = 0.01
   */

  def setRho(value: Double): this.type = {
    this.rho = value
    this
  }

  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    val stepFunction = this.stepSizeFunction match {
      case Some(func) => func
      case None => if(regParam > 0) stepSizeFunctionWithReg else stepSizeFunctionNoReg
    }
    val (weights, _) = ADMM.run(
      data,
      gradient,
      updater,
      regularizationOptimizer,
      stepSize,
      stoppingEpsilon,
      rho,
      stepFunction,
      numIterations,
      regParam,
      initialWeights)
    weights
  }
}

/*
 * run distributed admm
 * return: weights, loss in each iteration
 */

object ADMM extends Logging {
  private val MAX_LOSSES_TO_REPORT:Int = 20

  def run(
                       data: RDD[(Double, Vector)],
                       gradient: Gradient,
                       updater: DistUpdater,
                       regularizationOptimizer: RegularizationOptimizationDistributed,
                       stepSize: Double,
                       stoppingEpsilon: Double,
                       rho: Double,
                       stepSizeFunction: (Int) => Double,
                       numIterations: Int,
                       regParam: Double,
                       initialWeights: Vector): (Vector, Array[Double]) = {

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)

    // these are the final weights; in the algorithm they are the average weights across all partitions
    var weights = Vectors.dense(initialWeights.toArray)

    val numberOfFeatures = weights.size
    val noPartitions = data.partitions.length
    val zeroVector = Vectors.zeros(numberOfFeatures)

    var actualIterations = 1

    // number of records per partition
    val noRecordsPerPartition = Array.fill[Int](noPartitions)(0)
    data.mapPartitionsWithIndex{ case(idx,iter) =>
      List((idx,iter.length)).iterator
    }.map{ t => (t._1,t._2) }.collect().foreach{ case(idx,value) => noRecordsPerPartition(idx) = value }
    val bCastNoRecords = data.context.broadcast(noRecordsPerPartition)

    // array of penalties; one penalty vector per partition; we store them as single vector
    var penalties = Vectors.zeros(noPartitions*numberOfFeatures)
    // penalties corresponding to the regularization term
    var regularizationPenalties = Vectors.zeros(numberOfFeatures)

    /*
     * iterations
     */

    def runOneIteration(j:Int): Unit = {
      if( j<0 ) return

      // solve the problem for each partition
      val bcRegPenalties = data.context.broadcast(regularizationPenalties)
      val bcPenalties = data.context.broadcast(penalties)
      val bcWeights = data.context.broadcast(weights)

      val oneIterRdd = data.mapPartitionsWithIndex{ case (idx,iter) =>
        var w = bcWeights.value.copy
        val averageWeight = bcWeights.value
        val noRecords = bCastNoRecords.value(idx)
        val factor = rho/noRecords
        val penalties = bcPenalties.value
        val regPenalties = bcRegPenalties.value
        var iterCount = 1
        var loss = 0.0
        // gradient = loss gradient + rho/noRecords * (w + penalties(idx) - regPenalties)
        while(iter.hasNext) {
          val (label,features) = iter.next()
          // gradient
          val (newGradient,_) = gradient.compute(features, label, w)
          val (_,newLoss) = gradient.compute(features,label,averageWeight)
          // adjust with penalties
          BLAS.getInstance().daxpy(numberOfFeatures,factor,penalties.asInstanceOf[DenseVector].values,numberOfFeatures*idx,1,newGradient.asInstanceOf[DenseVector].values,0,1)
          // adjust for w
          BLAS.getInstance().daxpy(numberOfFeatures,factor,w.asInstanceOf[DenseVector].values,1,newGradient.asInstanceOf[DenseVector].values,1)
          // adjust for regularization penalty
          BLAS.getInstance().daxpy(numberOfFeatures,-factor,regPenalties.asInstanceOf[DenseVector].values,1,newGradient.asInstanceOf[DenseVector].values,1)
          // update current point; note that regularization in this case must be excluded
          val (w1,_) = updater.compute(w, newGradient, stepSize, stepSizeFunction, iterCount, 0.0)
          loss += newLoss
          w = w1
          iterCount += 1
        }
        List((w,loss,idx)).iterator
      }

      // compute average weights (the new current solution)
      val (sumWeight,totalLoss) = oneIterRdd.map{ t => (t._1,t._2) }.treeAggregate((Vectors.zeros(numberOfFeatures), 0.0))(
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

      // divide the sum to get the average
      BLAS.getInstance().dscal(numberOfFeatures,1.0/noPartitions,sumWeight.asInstanceOf[DenseVector].values,1)

      // calculate averagePenalties
      val averagePenalties = Vectors.zeros(numberOfFeatures)
      (0 until numberOfFeatures).toArray.foreach{ idx =>
        averagePenalties.asInstanceOf[DenseVector].values(idx) = BLAS.getInstance().dasum(noPartitions,penalties.asInstanceOf[DenseVector].values,idx,numberOfFeatures)
        averagePenalties.asInstanceOf[DenseVector].values(idx) /= noPartitions
      }

      // compute new regPenalties
      // solve the regularization subproblem: reg(x) + noPartitions * rho/2 ||x - sumWeights - regPenalty||
      val sumAll = new Array[Double](sumWeight.size)
      BLAS.getInstance().dcopy(sumWeight.size,sumWeight.asInstanceOf[DenseVector].values,1,sumAll,1)
      BLAS.getInstance().daxpy(sumWeight.size,1.0,averagePenalties.asInstanceOf[DenseVector].values,1,sumAll,1)
      regularizationPenalties = regularizationOptimizer.compute(zeroVector,weights.asInstanceOf[DenseVector],rho*noPartitions,regParam).asInstanceOf[DenseVector]

      // update penalties
      val bcNewRegularizationPenalties = data.context.broadcast(regularizationPenalties)
      // update penalties on each record in the rdd
      val pen = oneIterRdd.map{ case(partitionWeight,_,idx) =>
        val p = bcPenalties.value.copy
        val regPen = bcNewRegularizationPenalties.value
        // penalty = penalty + partitionWeigh - regularizationPenalties
        BLAS.getInstance().daxpy(regPen.size, 1.0, partitionWeight.asInstanceOf[DenseVector].values, 0, 1, p.asInstanceOf[DenseVector].values, numberOfFeatures*idx, 1)
        BLAS.getInstance().daxpy(regPen.size, -1.0, regPen.asInstanceOf[DenseVector].values, 0, 1, p.asInstanceOf[DenseVector].values, numberOfFeatures*idx, 1)
        // set all other values in p to be zero; needed for later doing the sum
        BLAS.getInstance().dscal(numberOfFeatures*idx,0.0,p.asInstanceOf[DenseVector].values,1)
        BLAS.getInstance().dscal(p.size-numberOfFeatures*(idx+1),0.0,p.asInstanceOf[DenseVector].values,numberOfFeatures*(idx+1),1)
        p
      }

      // we need to gather all of them (we sum them)
      penalties = pen.treeAggregate(Vectors.zeros(penalties.size))(
        seqOp = (c, v) => {
          BLAS.getInstance().daxpy(c.size,1.0,c.asInstanceOf[DenseVector].values,1,v.asInstanceOf[DenseVector].values,1)
          v
        },
        combOp = (c1, c2) => {
          BLAS.getInstance().daxpy(c1.size,1.0,c1.asInstanceOf[DenseVector].values,1,c2.asInstanceOf[DenseVector].values,1)
          c2
        }
      )

      // check if the previous vector weights and the new one sumWeight differ by more then stoppingEspilon
      val diffArray = new Array[Double](sumWeight.size)
      BLAS.getInstance().dcopy(sumWeight.size,sumWeight.asInstanceOf[DenseVector].values,1,diffArray,1)
      BLAS.getInstance().daxpy(sumWeight.size,-1.0,weights.asInstanceOf[DenseVector].values,1,diffArray,1)
      val normDiff = BLAS.getInstance().dnrm2(sumWeight.size,diffArray,1)

      weights = sumWeight

      // to compute the regularization value
      val regVal = updater.compute(weights, zeroVector, 0, x => x, 1, regParam)._2

      stochasticLossHistory.append(totalLoss+regVal)

      actualIterations += 1

      val stop = if(normDiff < stoppingEpsilon) true else false
      if(!stop) runOneIteration(j-1)
    }

    runOneIteration(numIterations-1)

    val noLossesToReport = math.min(MAX_LOSSES_TO_REPORT,actualIterations)
    logInfo("ADMM finished. Last %d losses %s".format(noLossesToReport,stochasticLossHistory.takeRight(noLossesToReport).mkString(", ")))

    (weights,stochasticLossHistory.toArray)
  }
}
