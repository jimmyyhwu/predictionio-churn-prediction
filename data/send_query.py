'''
Send sample query to churn prediction engine
'''

import predictionio
engine_client = predictionio.EngineClient(url="http://localhost:8000")
print engine_client.send_query({
	'state': 'KS',
  'account_length': 128,
  'area_code': 415,
  'international_plan': False,
  'voice_mail_plan': True,
  'number_vmail_messages': 25,
  'total_day_minutes': 265.1,
  'total_day_calls': 110,
  'total_day_charge': 45.07,
  'total_eve_minutes': 197.4,
  'total_eve_calls': 99,
  'total_eve_charge': 16.78,
  'total_night_minutes': 244.7,
  'total_night_calls': 91,
  'total_night_charge': 11.01,
  'total_intl_minutes': 10.0,
  'total_intl_calls': 3,
  'total_intl_charge': 2.7,
  'customer_service_calls': 1
})

