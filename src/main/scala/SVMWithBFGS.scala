import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.optimization.{HingeGradient,LBFGS, SquaredL2Updater}
import org.apache.spark.mllib.regression.GeneralizedLinearAlgorithm
import org.apache.spark.mllib.util.DataValidators

/**
 * Created by diego on 4/12/15.
 */

class SVMWithBFGS extends GeneralizedLinearAlgorithm[SVMModel] with Serializable {
  override val optimizer = new LBFGS(new HingeGradient, new SquaredL2Updater)

  override protected val validators = List(DataValidators.binaryLabelValidator)

  override protected def createModel(weights: Vector, intercept: Double) = {
    new SVMModel(weights, intercept)
  }
}
