package org.template.vanilla

import io.prediction.controller.IEngineFactory
import io.prediction.controller.Engine

case class Query(
  sepal_length: Double,
  sepal_width: Double,
  petal_length: Double,
  petal_width: Double
) extends Serializable

case class PredictedResult(
  val label: Array[Double]
) extends Serializable

object VanillaEngine extends IEngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("algo" -> classOf[Algorithm]),
      classOf[Serving])
  }
}

