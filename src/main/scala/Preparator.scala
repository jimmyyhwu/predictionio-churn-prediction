package org.template.classification

import io.prediction.controller.PPreparator
import org.apache.spark.SparkContext
import org.nd4j.linalg.dataset.DataSet

class Preparator
  extends PPreparator[TrainingData, PreparedData] {

  def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {
    new PreparedData(trainingData.data)
  }
}

class PreparedData(
  val data: DataSet
) extends Serializable

