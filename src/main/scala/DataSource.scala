package org.template.vanilla

import io.prediction.controller.PDataSource
import io.prediction.controller.EmptyEvaluationInfo
import io.prediction.controller.EmptyActualResult
import io.prediction.controller.Params
import io.prediction.data.storage.{PropertyMap, Event, Storage}

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import grizzled.slf4j.Logger
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j

case class DataSourceParams(appId: Int) extends Params

class DataSource(val dsp: DataSourceParams)
  extends PDataSource[TrainingData,
      EmptyEvaluationInfo, Query, EmptyActualResult] {

  @transient lazy val logger = Logger[this.type]

  override
  def readTraining(sc: SparkContext): TrainingData = {
    val eventsDb = Storage.getPEvents()
    val events: Array[(String, PropertyMap)] = eventsDb.aggregateProperties(
      appId = dsp.appId,
      entityType = "record",
      required = Some(List("sepal-length", "sepal-width", "petal-length", "petal-width", "species")))(sc).collect

    val features: INDArray = Nd4j.zeros(events.length.toInt, 4)
    val labels: INDArray = Nd4j.zeros(events.length.toInt, 3)

    events.zipWithIndex.foreach { case ((entityId, properties), row) =>
      val feature = Nd4j.create(
        Array(properties.get[Double]("sepal-length"),
          properties.get[Double]("sepal-width"),
          properties.get[Double]("petal-length"),
          properties.get[Double]("petal-width")
        )
      )
      features.putRow(row.toInt, feature)
      val label = Nd4j.create(
        properties.get[String]("species") match {
          case "Iris-setosa" => Array(1.0, 0.0, 0.0)
          case "Iris-versicolor" => Array(0.0, 1.0, 0.0)
          case "Iris-virginica" => Array(0.0, 0.0, 1.0)
        }
      )
      labels.putRow(row.toInt, label)
    }

    val data = new DataSet(features, labels)
    data.normalizeZeroMeanZeroUnitVariance
    new TrainingData(data)
  }
}

class TrainingData(
                    val data: DataSet
                    ) extends Serializable {
  override def toString = {
    s"events: [${data.numExamples()}] (${data.get(0)}...)"
  }
}
