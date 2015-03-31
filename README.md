# PredictionIO Classification Engine Template with MultiLayerNetwork from Deeplearning4j

## Engine Quickstart

### Overview

This engine template integrates the MultiLayerNetwork implementation from the Deeplearning4j library into PredictionIO. In this template, we use PredictionIO to classify the widely-known IRIS flower dataset by constructing a deep-belief net.

### Importing the IRIS Flower Dataset

The IRIS flower dataset contains 3 classes of flowers with 50 samples of each, for a total of 150 samples. An in-depth description can be found [here](https://archive.ics.uci.edu/ml/datasets/Iris) and a copy of the dataset can be found [here](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data). Each sample consists of four measurements (sepal length, sepal width, petal length, and petal width), as well as the class name (Iris-setosa, Iris-versicolor, and Iris-virginica).

A Python script has been provided to import the data to the event server using PredictionIO's Python SDK. Run the following command, replacing the access_key parameter with the value of your access key.

    python data/import_eventserver.py --access_key 4H5MigBfIvatrPKHNMkZSgKma58wRyy0Sfly20bLYLTSEZbrZ0uodTEBzqZukbcT

The data is stored in the event server with five properties per record. The measurements `sepal-length`, `sepal-width`, `petal-length`, and `petal-width` are stored as floats, and the `species` is stored as a string.

### Building, Training, and Deploying the Engine

Build the engine using the following command. The no-asm flag will instruct the PredictionIO to skip building external dependencies.

    pio build --no-asm

Train the model with the following command. Note that the commons-math3 library included with Spark is incompatible with the one required by Deeplearning4j, so we must specify that PredictionIO use the version located at lib/commons-math3-3.3.jar.

    pio train -- --driver-class-path lib/commons-math3-3.3.jar

Deploy the engine with the following command.

    pio deploy

### Querying the Engine

A query to the engine is a JSON consisting of the four flower measurements. The following is a sample JSON query.

    {'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2}

A Python script using the Python SDK has been provided to demonstrate a sample query. Execute the script as follows.

    python data/send_query.py

The model will return in a JSON response the predicted species based on the flower measurements in the query. The following is a sample JSON response.

    {'species': 'Iris-versicolor'}
