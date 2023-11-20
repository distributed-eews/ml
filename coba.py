import requests
import json

SERVICE_URL_REGISTER = "http://localhost:3000/restart"

station_code = "AGL"
print(f"Registering station {station_code}")
response = requests.post(SERVICE_URL_REGISTER,
          data=json.dumps({'station_code': station_code}),
          headers={"content-type": "application/json"})

print(response.text)