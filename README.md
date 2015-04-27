# PredictionIO Churn Prediction with MultiLayerNetwork from Deeplearning4j

## Engine Quickstart

### Importing the Orange churn dataset

A Python script has been provided to import the data to the event server using PredictionIO's Python SDK. Run the following command, replacing the access_key parameter with the value of your access key.

    python data/import_eventserver.py --access_key 4H5MigBfIvatrPKHNMkZSgKma58wRyy0Sfly20bLYLTSEZbrZ0uodTEBzqZukbcT

### Building, Training, and Deploying the Engine

Build the engine using the following command.

    pio build

Train the model with the following command. Note that the commons-math3 library included with Spark is incompatible with the one required by Deeplearning4j, so we must specify that PredictionIO use the version located at lib-commons-math3/commons-math3-3.3.jar.

    pio train -- --driver-class-path lib-commons-math3/commons-math3-3.3.jar

Deploy the engine with the following command.

    pio deploy
