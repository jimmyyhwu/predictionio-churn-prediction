package org.template.vanilla

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
  // extends PAlgorithm if Model contains RDD[]
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): Model = {
    val gen = new MersenneTwister(123)
    val conf = new NeuralNetConfiguration.Builder().iterations(100)
      .layerFactory(new PretrainLayerFactory(classOf[RBM]))
      .weightInit(WeightInit.SIZE)
      /*.dist(Nd4j.getDistributions.createNormal(1e-5, 1))*/
      .dist(new NormalDistribution(1e-5, 1))
      /*.activationFunction("tanh")*/
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
      .nIn(4)
      .nOut(3)
      .list(2)
      .useDropConnect(false)
      /*.hiddenLayerSizes(Array(3))*/
      .hiddenLayerSizes(3)
       /*.override(new ClassifierOverride(1))*/
      .build()
    val dbn = new MultiLayerNetwork(conf)
    data.data.normalizeZeroMeanZeroUnitVariance()
    data.data.shuffle()
    //val testAndTrain = next.splitTestAndTrain(110)
    //val train = testAndTrain.getTrain
    //d.fit(train)
    dbn.fit(data.data)

    //val test = testAndTrain.getTest
    //val eval = new Evaluation()
    //val output = d.output(test.getFeatureMatrix)
    //eval.eval(test.getLabels, output)
    val eval: Evaluation = new Evaluation
    val output: INDArray = dbn.output(data.data.getFeatureMatrix)
    eval.eval(data.data.getLabels, output)
    logger.info("Score " + eval.stats)

    new Model(dbn)
  }

  def predict(model: Model, query: Query): PredictedResult = {
    val features = Array(
      query.sepal_length,
      query.sepal_width,
      query.petal_length,
      query.petal_width
    )
    logger.info(Nd4j.create(features))
    val output = model.dbn.output(Nd4j.create(features))
    PredictedResult(Array(
      output.getDouble(0),
      output.getDouble(1),
      output.getDouble(2)
    ))
  }
}

class Model(val dbn: MultiLayerNetwork) extends Serializable {
  override def toString = s"dbn=${dbn}"
}

