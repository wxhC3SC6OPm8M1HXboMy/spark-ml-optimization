import org.apache.spark.mllib.optimization.Updater
import breeze.linalg.{norm => brzNorm, axpy => brzAxpy, Vector => BV}
import org.apache.spark.mllib.linalg.{Vector}

import optimization.BreezeHelper

/**
 * Created by diego on 4/19/15.
 */

class MySquaredL2Updater extends Updater {
  override def compute(
                        weightsOld: Vector,
                        gradient: Vector,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (Vector, Double) = {
    val thisIterStepSize = stepSize / (1+math.sqrt(iter))
    val brzWeights: BV[Double] = BreezeHelper.toBreeze(weightsOld).toDenseVector
    brzWeights :*= (1.0 - thisIterStepSize * regParam)
    brzAxpy(-thisIterStepSize, BreezeHelper.toBreeze(gradient), brzWeights)
    val norm = brzNorm(brzWeights, 2.0)

    (BreezeHelper.fromBreeze(brzWeights), 0.5 * regParam * norm * norm)
  }
}

