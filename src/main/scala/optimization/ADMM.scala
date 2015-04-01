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

class ADMM(private var gradient: Gradient, private var updater: DistUpdater) extends Distributed(gradient,updater) {
  private var rho: Double = 0.0001

  /*
   * set rho in ADMM; default = 0.0001
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
    val (weights, _) = ADMM.runADMM(
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
 * run distributed admm
 * return: weights, loss in each iteration
 * loss = sum of losses for each record based on current weight at that local iteration + regularization value of the weight for the next iteration
 */

object ADMM extends Logging {
  private val MAX_LOSSES_TO_REPORT:Int = 20

  def runADMM(
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
      while(iter.hasNext) {
        val (label,features) = iter.next()
        // gradient
        val (newGradient,_) = gradient.compute(features, label, w)
        val (_,newLoss) = gradient.compute(features,label,originalWeight)
        // update current point
        val (w1,_) = updater.compute(w, newGradient, stepSize, stepSizeFunction, iterCount, regParam)
        loss += newLoss
        w = w1
        iterCount += 1
      }
      List((w,loss,iterCount-1,idx)).iterator
    }

    /*
     * iterations
     */

    // broadcast number of records per partition
    val noRecordsPerPartition = Array.fill[Int](noPartitions)(0)
    oneIterRdd.map{ t => (t._4,t._3) }.collect().foreach{ case(idx,value) => noRecordsPerPartition(idx) = value }
    val bCastNoRecords = data.context.broadcast(noRecordsPerPartition)
    // array of penalties; one penalty vector per partition; we store them as single vector
    var penalties = Vectors.zeros(noPartitions*numberOfFeatures)
    // penalties corresponding to the regularization term
    var regularizationPenalties = Vectors.zeros(numberOfFeatures)
    // solution corresponding to the regularization term; it is obtained explicitly
    var regularizationWeights = Vectors.zeros(numberOfFeatures)

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

      // adjust the average to account for the regularization term
      BLAS.getInstance().daxpy(numberOfFeatures,1.0,regularizationWeights.asInstanceOf[DenseVector].values, 1, sumWeight.asInstanceOf[DenseVector].values,1)
      // divide the sum to get the average
      BLAS.getInstance().dscal(numberOfFeatures,1.0/(noPartitions+1),sumWeight.asInstanceOf[DenseVector].values,1)

      // check if the previous vector weights and the new one sumWeight differ by more then stoppingEspilon
      val diffArray = new Array[Double](sumWeight.size)
      BLAS.getInstance().dcopy(sumWeight.size,sumWeight.asInstanceOf[DenseVector].values,1,diffArray,1)
      BLAS.getInstance().daxpy(sumWeight.size,-1.0,weights.asInstanceOf[DenseVector].values,1,diffArray,1)
      val normDiff = BLAS.getInstance().dnrm2(sumWeight.size,diffArray,1)

      weights = sumWeight

      // to compute the regularization value
      val regVal = updater.compute(weights, zeroVector, 0, x => x, 1, regParam)._2

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

      // update penalty for regularization term
      BLAS.getInstance().daxpy(numberOfFeatures,rho,regularizationWeights.asInstanceOf[DenseVector].values,1,regularizationPenalties.asInstanceOf[DenseVector].values,1)
      BLAS.getInstance().daxpy(numberOfFeatures,-rho,weights.asInstanceOf[DenseVector].values,1,regularizationPenalties.asInstanceOf[DenseVector].values,1)

      /*
       * compute new weights for each partition
       */

      val bcNewPenalties = data.context.broadcast(penalties)

      val oneIterRdd = data.mapPartitionsWithIndex{ case (idx,iter) =>
        var w = bcWeights.value.copy
        val averageWeight = bcWeights.value
        val noRecords = bCastNoRecords.value(idx)
        val factor = rho/noRecords
        val penalties = bcNewPenalties.value
        var iterCount = 1
        var loss = 0.0
        // gradient = loss gradient + penalties/noRecords + rho/noRecords * (w - averageWeight)
        while(iter.hasNext) {
          val (label,features) = iter.next()
          // gradient
          val (newGradient,_) = gradient.compute(features, label, w)
          val (_,newLoss) = gradient.compute(features,label,averageWeight)
          // adjust with penalties
          BLAS.getInstance().daxpy(numberOfFeatures,1.0/noRecords,penalties.asInstanceOf[DenseVector].values,numberOfFeatures*idx,1,newGradient.asInstanceOf[DenseVector].values,0,1)
          // adjust for the averaging factor
          BLAS.getInstance().daxpy(numberOfFeatures,factor,w.asInstanceOf[DenseVector].values,1,newGradient.asInstanceOf[DenseVector].values,1)
          BLAS.getInstance().daxpy(numberOfFeatures,factor,averageWeight.asInstanceOf[DenseVector].values,1,newGradient.asInstanceOf[DenseVector].values,1)
          // update current point; note that regularization in this case must be excluded
          val (w1,_) = updater.compute(w, newGradient, stepSize, stepSizeFunction, iterCount, 0.0)
          loss += newLoss
          w = w1
          iterCount += 1
        }
        List((w,loss,idx)).iterator
      }

      actualIterations += 1

      // compute regularization solution; explicit formula = (rho * average - regularization penalty)/(rho + regParam)
      regularizationWeights = Vectors.zeros(numberOfFeatures)
      BLAS.getInstance().daxpy(numberOfFeatures,-1.0/(rho+regParam),regularizationPenalties.asInstanceOf[DenseVector].values,1,regularizationWeights.asInstanceOf[DenseVector].values,1)
      BLAS.getInstance().daxpy(numberOfFeatures,rho/(rho+regParam),weights.asInstanceOf[DenseVector].values,1,regularizationWeights.asInstanceOf[DenseVector].values,1)

      runOneIteration(j-1,Some(oneIterRdd),stop)
    }

    runOneIteration(numIterations-2,Some(oneIterRdd.map{ t => (t._1,t._2,t._4) }),stopFlag = false)

    val noLossesToReport = math.min(MAX_LOSSES_TO_REPORT,actualIterations)
    logInfo("ADMM finished. Last %d losses %s".format(noLossesToReport,stochasticLossHistory.takeRight(noLossesToReport).mkString(", ")))

    (weights,stochasticLossHistory.toArray)
  }
}
