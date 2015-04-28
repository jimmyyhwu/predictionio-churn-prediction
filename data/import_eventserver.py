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
  prop_names = [
    'state',
    'account_length',
    'area_code',
    'international_plan',
    'voice_mail_plan',
    'number_vmail_messages',
    'total_day_minutes',
    'total_day_calls',
    'total_day_charge',
    'total_eve_minutes',
    'total_eve_calls',
    'total_eve_charge',
    'total_night_minutes',
    'total_night_calls',
    'total_night_charge',
    'total_intl_minutes',
    'total_intl_calls',
    'total_intl_charge',
    'customer_service_calls',
    'churn'
  ]
  for line in f:
    data = line.strip().split(',')
    props = {}
    for i, prop in enumerate(data):
      if i in [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
        props[prop_names[i]] = float(prop)
      elif i in [3, 4, 19]:
        if prop == 'True' or prop == 'Yes':
          props[prop_names[i]] = True
        elif prop == 'False' or prop == 'No':
          props[prop_names[i]] = False
      elif i in [0, 2]:
        props[prop_names[i]] = prop
    
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
