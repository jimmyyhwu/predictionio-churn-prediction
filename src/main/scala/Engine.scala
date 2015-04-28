package org.template.classification

import io.prediction.controller.IEngineFactory
import io.prediction.controller.Engine

case class Query(
  state: String,
  account_length: Double,
  area_code: String,
  international_plan: Boolean,
  voice_mail_plan: Boolean,
  number_vmail_messages: Double,
  total_day_minutes: Double,
  total_day_calls: Double,
  total_day_charge: Double,
  total_eve_minutes: Double,
  total_eve_calls: Double,
  total_eve_charge: Double,
  total_night_minutes: Double,
  total_night_calls: Double,
  total_night_charge: Double,
  total_intl_minutes: Double,
  total_intl_calls: Double,
  total_intl_charge: Double,
  customer_service_calls: Double
) extends Serializable

case class PredictedResult(
  val churn: String
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

