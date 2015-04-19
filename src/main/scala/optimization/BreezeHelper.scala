package optimization

import org.apache.spark.mllib.linalg.{SparseVector, DenseVector, Vector}
import breeze.linalg.{Vector => BV, DenseVector => BDV, SparseVector => BSV}

/**
 * Created by diego on 4/19/15.
 */

/*
 * needed because to/from Breeze are private in Vector
 */

object BreezeHelper {
  def toBreeze(v:Vector):BV[Double] = {
    v match {
      case value:DenseVector => new BDV[Double](value.values)
      case value:SparseVector => new BSV[Double](value.indices, value.values, value.size)
    }
  }

  def fromBreeze(breezeVector: BV[Double]): Vector = {
    breezeVector match {
      case v: BDV[Double] =>
        if (v.offset == 0 && v.stride == 1 && v.length == v.data.length) {
          new DenseVector(v.data)
        } else {
          new DenseVector(v.toArray)
        }
      case v: BSV[Double] =>
        if (v.index.length == v.used) {
          new SparseVector(v.length, v.index, v.data)
        } else {
          new SparseVector(v.length, v.index.slice(0, v.used), v.data.slice(0, v.used))
        }
      case v: BV[_] =>
        sys.error("Unsupported Breeze vector type: " + v.getClass.getName)
    }
  }
}
