package optimization

import com.github.fommil.netlib.BLAS

import org.apache.spark.mllib.linalg.{Vectors, Vector, DenseVector}

/**
 *  Solves the regularization subproblem for distributed problems.
 *  min_w regParam * reg(w) + penalty*w + rho/2 pow(|| w-aveWeight ||,2)
 */

abstract class RegularizationOptimizationDistributed extends Serializable {
  /**
    * solves: min_w regParam * reg(w) + penalty*w + rho/2 pow(|| w-aveWeight ||,2)
    * 
    *  @param penalties - Column matrix of size dx1 where d is the number of features.
    *  @param aveWeights - Column matrix of size dx1 where d is the number of features.
    *  @param rho - rho value
    *  @param regParam - Regularization parameter
    * 
    *  @return A column matrix of size dx1 where d is the number of features
    */
  def compute(
      penalties: Vector,
      aveWeights: Vector,
      rho: Double,
      regParam: Double): Vector
}

/**
 * For the L2 norm the solution to the problem is explicitly given as (easy to derive by setting the first derivative to zero:
 * (rho * aveWeights - penalties)/(regParam + rho)
 */

class L2RegularizationOptimizationDistributed extends RegularizationOptimizationDistributed {
  override def compute(
                        penalties: Vector,
                        aveWeights: Vector,
                        rho: Double,
                        regParam: Double): Vector = {
    val numberOfFeatures = penalties.size
    val regularizationWeights = Vectors.zeros(numberOfFeatures)
    BLAS.getInstance().daxpy(numberOfFeatures,-1.0/(rho+regParam),penalties.asInstanceOf[DenseVector].values,1,regularizationWeights.asInstanceOf[DenseVector].values,1)
    BLAS.getInstance().daxpy(numberOfFeatures,rho/(rho+regParam),aveWeights.asInstanceOf[DenseVector].values,1,regularizationWeights.asInstanceOf[DenseVector].values,1)

    regularizationWeights
  }
}

