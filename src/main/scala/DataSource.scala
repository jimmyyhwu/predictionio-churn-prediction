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
      required = Some(List(
        "state",
        "account_length",
        "area_code",
        "international_plan",
        "voice_mail_plan",
        "number_vmail_messages",
        "total_day_minutes",
        "total_day_calls",
        "total_day_charge",
        "total_eve_minutes",
        "total_eve_calls",
        "total_eve_charge",
        "total_night_minutes",
        "total_night_calls",
        "total_night_charge",
        "total_intl_minutes",
        "total_intl_calls",
        "total_intl_charge",
        "customer_service_calls",
        "churn"
      ))
    )(sc).collect()

    val features: INDArray = Nd4j.zeros(events.length, 15+51+2+2)
    val labels: INDArray = Nd4j.zeros(events.length, 2)

    events.zipWithIndex.foreach { case ((entityId, properties), row) =>
      val continuousFeatures = Array(
        properties.get[Double]("account_length"),
        properties.get[Double]("number_vmail_messages"),
        properties.get[Double]("total_day_minutes"),
        properties.get[Double]("total_day_calls"),
        properties.get[Double]("total_day_charge"),
        properties.get[Double]("total_eve_minutes"),
        properties.get[Double]("total_eve_calls"),
        properties.get[Double]("total_eve_charge"),
        properties.get[Double]("total_night_minutes"),
        properties.get[Double]("total_night_calls"),
        properties.get[Double]("total_night_charge"),
        properties.get[Double]("total_intl_minutes"),
        properties.get[Double]("total_intl_calls"),
        properties.get[Double]("total_intl_charge"),
        properties.get[Double]("customer_service_calls")
      )

      val states = List(
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
      )
      val state = Array.fill[Double](51)(0.0)

      val index = states.indexOf(properties.get[String]("state"))
      if (index > -1) {
        state(index) = 1.0
      } else {
        state(50) = 1.0
      }

      val intPlan = properties.get[Boolean]("international_plan") match {
        case true => Array(1.0, 0.0)
        case false => Array(0.0, 1.0)
      }
      val vmailPlan = properties.get[Boolean]("voice_mail_plan") match {
        case true => Array(1.0, 0.0)
        case false => Array(0.0, 1.0)
      }

      val user = Nd4j.create(continuousFeatures ++ state ++ intPlan ++ vmailPlan)
      features.putRow(row.toInt, user)
      val label = Nd4j.create(
        properties.get[Boolean]("churn") match {
          case true => Array(1.0, 0.0)
          case false => Array(0.0, 1.0)
        }
      )
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
