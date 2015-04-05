package classification

import org.apache.spark.mllib.regression.GeneralizedLinearAlgorithm
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.optimization.HingeGradient
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.mllib.linalg.Vector

import optimization.{IPA,ADMM,PH,PHDistributeRegularizationTerm}
import optimization.{L2RegularizationOptimizationDistributed,SquaredL2Updater}

/**
 * Created by diego on 1/31/15.
 * SVM with distributed optimization variants
 */

class SVMWithIPA extends GeneralizedLinearAlgorithm[SVMModel] with Serializable {
    override val optimizer = new IPA(new HingeGradient, new SquaredL2Updater)

    override protected val validators = List(DataValidators.binaryLabelValidator)

    override protected def createModel(weights: Vector, intercept: Double) = {
      new SVMModel(weights, intercept)
    }
}

class SVMWithADMM extends GeneralizedLinearAlgorithm[SVMModel] with Serializable {
  override val optimizer = new ADMM(new HingeGradient, new SquaredL2Updater, new L2RegularizationOptimizationDistributed)

  override protected val validators = List(DataValidators.binaryLabelValidator)

  override protected def createModel(weights: Vector, intercept: Double) = {
    new SVMModel(weights, intercept)
  }
}

class SVMWithPH extends GeneralizedLinearAlgorithm[SVMModel] with Serializable {
  override val optimizer = new PH(new HingeGradient, new SquaredL2Updater, new L2RegularizationOptimizationDistributed)

  override protected val validators = List(DataValidators.binaryLabelValidator)

  override protected def createModel(weights: Vector, intercept: Double) = {
    new SVMModel(weights, intercept)
  }
}

class SVMWithPHDistributeRegularizationTerm extends GeneralizedLinearAlgorithm[SVMModel] with Serializable {
  override val optimizer = new PHDistributeRegularizationTerm(new HingeGradient, new SquaredL2Updater)

  override protected val validators = List(DataValidators.binaryLabelValidator)

  override protected def createModel(weights: Vector, intercept: Double) = {
    new SVMModel(weights, intercept)
  }
}
