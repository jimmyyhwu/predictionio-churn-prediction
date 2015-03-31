import AssemblyKeys._

assemblySettings

name := "predictionio-template-classification-dl4j-multilayer-network"

organization := "io.prediction"

libraryDependencies ++= Seq(
  "io.prediction"    %% "core"          % pioVersion.value % "provided",
  "org.apache.spark" %% "spark-core"    % "1.2.0" % "provided",
  "org.apache.spark" %% "spark-mllib"   % "1.2.0" % "provided",
  "org.nd4j" % "nd4j-jblas" % "0.0.3.5.5.2",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.0.3.3")
