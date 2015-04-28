package org.template.classification

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params
import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.SparkContext
import grizzled.slf4j.Logger
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.models.featuredetectors.rbm.RBM
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.layers.factory.PretrainLayerFactory
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.api.activation.Activations
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

case class AlgorithmParams(mult: Int) extends Params

class Algorithm(val ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): Model = {
    val gen = new MersenneTwister(123)
    val conf = new NeuralNetConfiguration.Builder().iterations(100)
      .layerFactory(new PretrainLayerFactory(classOf[RBM]))
      .weightInit(WeightInit.SIZE)
      .dist(new NormalDistribution(1e-5, 1))
      .activationFunction(Activations.tanh)
      .momentum(0.9)
      .dropOut(0.8)
      .optimizationAlgo(OptimizationAlgorithm.GRADIENT_DESCENT)
      .constrainGradientToUnitNorm(true)
      .k(5)
      .regularization(true)
      .l2(2e-4)
      .visibleUnit(RBM.VisibleUnit.GAUSSIAN)
      .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
      .lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
      .learningRate(1e-1f)
      .nIn(15+51+2+2)
      .nOut(2)
      .list(2)
      .useDropConnect(false)
      .hiddenLayerSizes(3)
      .build()
    val dbn = new MultiLayerNetwork(conf)
    dbn.fit(data.data)
    val eval = new Evaluation()
    val output = dbn.output(data.data.getFeatureMatrix())
    eval.eval(data.data.getLabels(),output)
    logger.info("Score " + eval.stats())
    new Model(dbn)
  }

  def predict(model: Model, query: Query): PredictedResult = {
    val continuousFeatures = Array(
      query.account_length,
      query.number_vmail_messages,
      query.total_day_minutes,
      query.total_day_calls,
      query.total_day_charge,
      query.total_eve_minutes,
      query.total_eve_calls,
      query.total_eve_charge,
      query.total_night_minutes,
      query.total_night_calls,
      query.total_night_charge,
      query.total_intl_minutes,
      query.total_intl_calls,
      query.total_intl_charge,
      query.customer_service_calls
    )

    val states = List(
      "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
      "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
      "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
      "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
      "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
    )
    val state = Array.fill[Double](51)(0.0)
    val index = states.indexOf(query.state)
    if (index > -1) {
      state(index) = 1.0
    } else {
      state(50) = 1.0
    }

    val intPlan = query.international_plan match {
      case true => Array(1.0, 0.0)
      case false => Array(0.0, 1.0)
    }
    val vmailPlan = query.voice_mail_plan match {
      case true => Array(1.0, 0.0)
      case false => Array(0.0, 1.0)
    }

    val output = model.dbn.predict(Nd4j.create(continuousFeatures ++ state ++ intPlan ++ vmailPlan))
    val labelNames = Array("True", "False")
    new PredictedResult(labelNames(output(0)))
  }
}

class Model(val dbn: MultiLayerNetwork) extends Serializable {
  override def toString = s"dbn=${dbn}"
}

