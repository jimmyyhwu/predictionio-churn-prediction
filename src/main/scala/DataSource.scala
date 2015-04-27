package org.template.classification

import io.prediction.controller.PDataSource
import io.prediction.controller.EmptyEvaluationInfo
import io.prediction.controller.EmptyActualResult
import io.prediction.controller.Params
import io.prediction.data.storage.{PropertyMap, Storage}
import org.apache.spark.SparkContext
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
      entityType = "user",
      required = Some(List(/*"State",*/
        "Account length",
        /*"Area code",*/
        "International plan",
        "Voice mail plan",
        "Number vmail messages",
        "Total day minutes",
        "Total day calls",
        "Total day charge",
        "Total eve minutes",
        "Total eve calls",
        "Total eve charge",
        "Total night minutes",
        "Total night calls",
        "Total night charge",
        "Total intl minutes",
        "Total intl calls",
        "Total intl charge",
        "Customer service calls",
        "Churn"))
    )(sc).collect()

    val features: INDArray = Nd4j.zeros(events.length, 17)
    val labels: INDArray = Nd4j.zeros(events.length, 1)

    events.zipWithIndex.foreach { case ((entityId, properties), row) =>
      val feature = Nd4j.create(
        Array(properties.get[Double]("Account length"),
          properties.get[Double]("International plan"),
          properties.get[Double]("Voice mail plan"),
          properties.get[Double]("Number vmail messages"),
          properties.get[Double]("Total day minutes"),
          properties.get[Double]("Total day calls"),
          properties.get[Double]("Total day charge"),
          properties.get[Double]("Total eve minutes"),
          properties.get[Double]("Total eve calls"),
          properties.get[Double]("Total eve charge"),
          properties.get[Double]("Total night minutes"),
          properties.get[Double]("Total night calls"),
          properties.get[Double]("Total night charge"),
          properties.get[Double]("Total intl minutes"),
          properties.get[Double]("Total intl calls"),
          properties.get[Double]("Total intl charge"),
          properties.get[Double]("Customer service calls")
        )
      )
      features.putRow(row.toInt, feature)
      val label = Nd4j.create(Array(properties.get[Double]("Churn")))
      labels.putRow(row.toInt, label)
    }

    val data = new DataSet(features, labels)
    data.normalizeZeroMeanZeroUnitVariance()
    data.shuffle()
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
