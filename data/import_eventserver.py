'''
Import orange churn data
'''

import predictionio
import argparse

def import_events(client, file):
  f = open(file, 'rb')
  next(f)
  count = 0
  print 'Importing data...'
  prop_names = ['State',
    'Account length',
    'Area code',
    'International plan',
    'Voice mail plan',
    'Number vmail messages',
    'Total day minutes',
    'Total day calls',
    'Total day charge',
    'Total eve minutes',
    'Total eve calls',
    'Total eve charge',
    'Total night minutes',
    'Total night calls',
    'Total night charge',
    'Total intl minutes',
    'Total intl calls',
    'Total intl charge',
    'Customer service calls',
    'Churn']
  for line in f:
    data = line.strip().split(',')
    props = {}
    for i, prop in enumerate(data):
      if i in [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
        props[prop_names[i]] = float(prop)
      elif i in [3, 4, 19]:
        if prop == 'True' or prop == 'Yes':
          props[prop_names[i]] = float(1)
        elif prop == 'False' or prop == 'No':
          props[prop_names[i]] = float(0)
      elif i in [0, 2]:
        props[prop_names[i]] = prop
      else:
        print i
    
    client.create_event(
      event = '$set',
      entity_type = 'user',
      entity_id = str(count),
      properties = props
    )
    count += 1
  f.close()
  print '%s records were imported.' % count

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Import orange churn data')
  parser.add_argument('--access_key', default='invalid_access_key')
  parser.add_argument('--url', default='http://localhost:7070')
  parser.add_argument('--file', default='./data/churn-orange-train.csv')

  args = parser.parse_args()
  print args

  client = predictionio.EventClient(
    access_key=args.access_key,
    url=args.url,
    threads=5,
    qsize=500)
  import_events(client, args.file)
