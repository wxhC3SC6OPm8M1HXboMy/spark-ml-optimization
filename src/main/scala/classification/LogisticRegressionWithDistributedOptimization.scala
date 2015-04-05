package classification

import org.apache.spark.mllib.regression.GeneralizedLinearAlgorithm
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.optimization.LogisticGradient
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.mllib.linalg.Vector

import optimization.{IPA,ADMM,PH,PHDistributeRegularizationTerm}
import optimization.{L2RegularizationOptimizationDistributed,SquaredL2Updater}

/**
 * Created by diego on 1/31/15.
 * Logistic regression with distributed optimization
 */

class LogisticRegressionWithIPA extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {
    override val optimizer = new IPA(new LogisticGradient, new SquaredL2Updater)

    override protected val validators = List(DataValidators.binaryLabelValidator)

    override protected def createModel(weights: Vector, intercept: Double) = {
      new LogisticRegressionModel(weights, intercept)
    }
}

class LogisticRegressionWithADMM extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {
  override val optimizer = new ADMM(new LogisticGradient, new SquaredL2Updater, new L2RegularizationOptimizationDistributed)

  override protected val validators = List(DataValidators.binaryLabelValidator)

  override protected def createModel(weights: Vector, intercept: Double) = {
    new LogisticRegressionModel(weights, intercept)
  }
}

class LogisticRegressionWithPH extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {
  override val optimizer = new PH(new LogisticGradient, new SquaredL2Updater, new L2RegularizationOptimizationDistributed)

  override protected val validators = List(DataValidators.binaryLabelValidator)

  override protected def createModel(weights: Vector, intercept: Double) = {
    new LogisticRegressionModel(weights, intercept)
  }
}

class LogisticRegressionWithPHDistributeRegularizationTerm extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {
  override val optimizer = new PHDistributeRegularizationTerm(new LogisticGradient, new SquaredL2Updater)

  override protected val validators = List(DataValidators.binaryLabelValidator)

  override protected def createModel(weights: Vector, intercept: Double) = {
    new LogisticRegressionModel(weights, intercept)
  }
}
