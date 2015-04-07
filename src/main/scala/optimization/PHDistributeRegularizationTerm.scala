package optimization

import com.github.fommil.netlib.BLAS

import org.apache.spark.mllib.optimization.Gradient
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vector,Vectors,DenseVector,SparseVector}
import org.apache.spark.mllib.rdd.RDDFunctions._

import scala.collection.mutable.ArrayBuffer

import optimization.{Updater=>DistUpdater}

/**
 * Created by diego on 1/28/15.
 *
 * Solves by using PH: sum_{partition i} [sum_{records r in i) loss(r,w)+ regParam/n * reg(w)]
 * Here n is the number of partitions.
 *
 * Each partition solves: sum_{records r in i) [loss(r,w)+ regParam/(n_i*n) * reg(w) + penalty*w/n_i + rho/2 * (w-average)^2/n_i]
 * n_i = the number of records in partition i
 * This problem is solved by single pass updating the gradient
 */

class PHDistributeRegularizationTerm(private var gradient: Gradient, private var updater: DistUpdater) extends Distributed(gradient,updater) {
  private var rho: Double = 0.01

  /*
   * set rho; default = 0.0001
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
    val (weights, _) = PHDistributeRegularizationTerm.run(
      data,
      gradient,
      updater,
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
 * run distributed ph where the regularization term is assumed to be distributed to each partition
 * return: weights, loss in each iteration
 */

object PHDistributeRegularizationTerm extends Logging {
  private val MAX_LOSSES_TO_REPORT:Int = 20

  def run(
                       data: RDD[(Double, Vector)],
                       gradient: Gradient,
                       updater: DistUpdater,
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

    /*
     * the first iteration: corresponds to regular single iteration of IPA
     */

    val bcInitialWeights = data.context.broadcast(weights)

    // for each partition in the rdd
    val oneIterRdd = data.mapPartitionsWithIndex{ case (idx,iter) =>
      var w = bcInitialWeights.value.copy
      val originalWeight = bcInitialWeights.value
      var iterCount = 1
      var loss = 0.0
      val adjustedRegParam = regParam/(bCastNoRecords.value(idx) * noPartitions)
      while(iter.hasNext) {
        val (label,features) = iter.next()
        // gradient
        val (newGradient,_) = gradient.compute(features, label, w)
        val (_,newLoss) = gradient.compute(features,label,originalWeight)
        // update current point
        val (w1,_) = updater.compute(w, newGradient, stepSize, stepSizeFunction, iterCount, adjustedRegParam)
        loss += newLoss
        w = w1
        iterCount += 1
      }
      List((w,loss,iterCount-1,idx)).iterator
    }

    /*
     * iterations
     */

    // array of penalties; one penalty vector per partition; we store them as single vector
    var penalties = Vectors.zeros(noPartitions*numberOfFeatures)

    // prevIterRdd: for each partition (record in the rdd) it has: solution (partition) weights,loss,partition id
    def runOneIteration(j:Int,_prevIterRdd:Option[RDD[(Vector,Double,Int)]],stopFlag: Boolean): Unit = {
      if (j < 0 || stopFlag) return

      /*
       * average solution
       */

      val prevIterRdd = _prevIterRdd match {
        case Some(x) => x
      }

      val (sumWeight,totalLoss) = prevIterRdd.map{ t => (t._1,t._2) }.treeAggregate((Vectors.zeros(numberOfFeatures), 0.0))(
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

      // check if the previous vector weights and the new one sumWeight differ by more then stoppingEspilon
      val diffArray = new Array[Double](sumWeight.size)
      BLAS.getInstance().dcopy(sumWeight.size,sumWeight.asInstanceOf[DenseVector].values,1,diffArray,1)
      BLAS.getInstance().daxpy(sumWeight.size,-1.0,weights.asInstanceOf[DenseVector].values,1,diffArray,1)
      val normDiff = BLAS.getInstance().dnrm2(sumWeight.size,diffArray,1)

      // to compute the regularization value
      val regVal = updater.compute(weights, zeroVector, 0, x => x, 1, regParam)._2

      weights = sumWeight

      stochasticLossHistory.append(totalLoss+regVal)

      val stop = if(normDiff < stoppingEpsilon) true else false
      if(stop) {
        runOneIteration(j-1,None,stop)
      }

      // broadcast weights; in a single iteration these are the average weights
      val bcWeights = data.context.broadcast(weights)
      val bcPenalties = data.context.broadcast(penalties)

      /*
       * update penalties
       */

      // update penalties on each record in the rdd
      val pen = prevIterRdd.map{ case(partitionWeight,_,idx) =>
        val w = bcWeights.value
        val p = bcPenalties.value.copy
        // penalty = penalty + rho(partitionWeight - w)
        BLAS.getInstance().daxpy(w.size, rho, partitionWeight.asInstanceOf[DenseVector].values, 0, 1, p.asInstanceOf[DenseVector].values, numberOfFeatures*idx, 1)
        BLAS.getInstance().daxpy(w.size, -rho, w.asInstanceOf[DenseVector].values, 0, 1, p.asInstanceOf[DenseVector].values, numberOfFeatures*idx, 1)
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

      bcPenalties.destroy()

      /*
       * compute new weights for each partition
       */

      val bcNewPenalties = data.context.broadcast(penalties)

      val oneIterRdd = data.mapPartitionsWithIndex{ case (idx,iter) =>
        var w = bcWeights.value.copy
        val averageWeight = bcWeights.value
        val noRecords = bCastNoRecords.value(idx)
        val factor = rho/noRecords
        val factorReg = regParam/(noPartitions*noRecords)
        val penalties = bcNewPenalties.value
        var iterCount = 1
        var loss = 0.0
        // gradient of loss(r,w)+ regParam/(n_i*n) * reg(w) + penalty*w/n_i + rho/2 * (w-average)^2/n_i
        // gradient = loss gradient + penalties/noRecords + rho/noRecords * (w - averageWeight) + gradient of reg(w) * factorReg
        while(iter.hasNext) {
          val (label,features) = iter.next()
          // gradient
          val (ng,_) = gradient.compute(features, label, w)
          val newGradient = ng match {
            case g: DenseVector => g
            case g: SparseVector => Vectors.dense(g.toArray)
          }
          val (_,newLoss) = gradient.compute(features,label,averageWeight)
          // adjust with penalties
          BLAS.getInstance().daxpy(numberOfFeatures,1.0/noRecords,penalties.asInstanceOf[DenseVector].values,numberOfFeatures*idx,1,newGradient.asInstanceOf[DenseVector].values,0,1)
          // adjust for the averaging factor
          BLAS.getInstance().daxpy(numberOfFeatures,factor,w.asInstanceOf[DenseVector].values,1,newGradient.asInstanceOf[DenseVector].values,1)
          BLAS.getInstance().daxpy(numberOfFeatures,-factor,averageWeight.asInstanceOf[DenseVector].values,1,newGradient.asInstanceOf[DenseVector].values,1)
          // update current point; note that regularization parameter in this case is factorReg (the gradient for regularization is taken in compute)
          val (w1,_) = updater.compute(w, newGradient, stepSize, stepSizeFunction, iterCount, factorReg)
          loss += newLoss
          w = w1
          iterCount += 1
        }
        List((w,loss,idx)).iterator
      }

      actualIterations += 1

      runOneIteration(j-1,Some(oneIterRdd),stop)
    }

    runOneIteration(numIterations-2,Some(oneIterRdd.map{ t => (t._1,t._2,t._4) }),stopFlag = false)

    val noLossesToReport = math.min(MAX_LOSSES_TO_REPORT,actualIterations)
    logInfo("PH Distributed Regularization finished. Last %d losses %s".format(noLossesToReport,stochasticLossHistory.takeRight(noLossesToReport).mkString(", ")))

    (weights,stochasticLossHistory.toArray)
  }
}
