package org.template.classification

import io.prediction.controller.IEngineFactory
import io.prediction.controller.Engine

case class Query(
  sepal_length: Double,
  sepal_width: Double,
  petal_length: Double,
  petal_width: Double
) extends Serializable

case class PredictedResult(
  val label: String
) extends Serializable

object ClassificationEngine extends IEngineFactory {
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("dbn" -> classOf[Algorithm]),
      classOf[Serving])
  }
}

