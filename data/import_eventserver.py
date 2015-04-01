'''
Import sample data for IRIS example
'''

import predictionio
import argparse

def import_events(client, file):
  f = open(file, 'rb')
  count = 0
  print 'Importing data...'
  for line in f:
    data = line.strip().split(',')
    client.create_event(
      event = '$set',
      entity_type = 'record',
      entity_id = str(count),
      properties = {
        'sepal-length': float(data[0]),
        'sepal-width': float(data[1]),
        'petal-length': float(data[2]),
        'petal-width': float(data[3]),
        'species': data[4]
      }
    )
    count += 1
  f.close()
  print '%s records were imported.' % count

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Import sample data for IRIS example')
  parser.add_argument('--access_key', default='invalid_access_key')
  parser.add_argument('--url', default='http://localhost:7070')
  parser.add_argument('--file', default='./data/iris.data')

  args = parser.parse_args()
  print args

  client = predictionio.EventClient(
    access_key=args.access_key,
    url=args.url,
    threads=5,
    qsize=500)
  import_events(client, args.file)
