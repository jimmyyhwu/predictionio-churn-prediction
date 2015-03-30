import AssemblyKeys._

assemblySettings

name := "template-scala-parallel-vanilla"

organization := "io.prediction"

libraryDependencies ++= Seq(
  "io.prediction"    %% "core"          % pioVersion.value % "provided",
  "org.apache.spark" %% "spark-core"    % "1.2.0" % "provided",
  "org.apache.spark" %% "spark-mllib"   % "1.2.0" % "provided",
  "org.nd4j" % "nd4j-api" % "0.0.3.5.5.2",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.0.3.3")
