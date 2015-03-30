package org.template.vanilla

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params
import org.apache.commons.math3.random.MersenneTwister

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

import grizzled.slf4j.Logger
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.models.featuredetectors.rbm.RBM
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.layers.factory.PretrainLayerFactory
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

case class AlgorithmParams(mult: Int) extends Params

class Algorithm(val ap: AlgorithmParams)
  // extends PAlgorithm if Model contains RDD[]
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): Model = {
    val gen = new MersenneTwister(123)
    val conf = new NeuralNetConfiguration.Builder().iterations(100)
      .layerFactory(new PretrainLayerFactory(classOf[RBM]))
      .weightInit(WeightInit.SIZE)
      .dist(Nd4j.getDistributions.createNormal(1e-5, 1))
      .activationFunction("tanh")
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
      .nIn(4)
      .nOut(3)
      .list(2)
      .useDropConnect(false)
      .hiddenLayerSizes(Array(3))
    .override(new ClassifierOverride(1))
      .build()
    val d = new MultiLayerNetwork(conf)
    val iter = new IrisDataSetIterator(150, 150)
    val next = iter.next()
    next.normalizeZeroMeanZeroUnitVariance()
    next.shuffle()
    val testAndTrain = next.splitTestAndTrain(110)
    val train = testAndTrain.getTrain
    d.fit(train)
    val test = testAndTrain.getTest
    val eval = new Evaluation()
    val output = d.output(test.getFeatureMatrix)
    eval.eval(test.getLabels, output)
    logger.info("Score " + eval.stats())



    // Simply count number of events
    // and multiple it by the algorithm parameter
    // and store the number as model
    val count = data.events.count().toInt * ap.mult
    new Model(mc = count)
  }

  def predict(model: Model, query: Query): PredictedResult = {
    // Prefix the query with the model data
    val result = s"${model.mc}-${query.q}"
    PredictedResult(p = result)
  }
}

class Model(val mc: Int) extends Serializable {
  override def toString = s"mc=${mc}"
}
