'''
Send sample query to classification engine
'''

import predictionio
engine_client = predictionio.EngineClient(url="http://localhost:8000")
print engine_client.send_query({'sepal_length': 5.1, 'sepal_width': 3.5, 'petal_length': 1.4, 'petal_width': 0.2})
