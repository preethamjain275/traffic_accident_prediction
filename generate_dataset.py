import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 10000

weather_options = ['Clear', 'Cloudy', 'Rain', 'Heavy Rain', 'Fog', 'Thunderstorm']
weather_weights = [0.40, 0.30, 0.15, 0.05, 0.05, 0.05]
road_types      = ['National Highway', 'State Highway', 'City Street', 'Rural Road', 'Ring Road']
road_weights    = [0.25, 0.25, 0.35, 0.05, 0.10]
locations       = ['Bengaluru, Karnataka', 'Mysuru, Karnataka', 'Mangaluru, Karnataka', 'Hubballi, Karnataka', 'Belagavi, Karnataka']

hours = np.random.choice(range(24), N, p=[
    0.02,0.01,0.01,0.01,0.02,0.03,
    0.05,0.07,0.06,0.05,0.05,0.05,
    0.05,0.05,0.05,0.05,0.06,0.07,
    0.06,0.05,0.04,0.04,0.03,0.02
])

weather      = np.random.choice(weather_options, N, p=weather_weights)
road_type    = np.random.choice(road_types, N, p=road_weights)
temperature  = np.clip(np.random.normal(28, 5, N), 15, 42)
wind_speed   = np.clip(np.random.exponential(12, N), 0, 50)
visibility   = np.clip(np.random.normal(10, 3, N), 0.1, 15)
humidity     = np.clip(np.random.normal(65, 15, N), 20, 100)
pressure     = np.clip(np.random.normal(1010, 5, N), 990, 1025)
precipitation = np.where(
    np.isin(weather, ['Rain', 'Heavy Rain', 'Thunderstorm']),
    np.clip(np.random.exponential(10, N), 0, 100.0), 0.0
)
speed_limit  = np.random.choice([30,40,50,60,80,100,120], N, p=[0.1,0.2,0.3,0.2,0.1,0.05,0.05])
junction     = np.random.choice([0,1], N, p=[0.6,0.4])
traffic_signal = np.random.choice([0,1], N, p=[0.5,0.5])
crossing     = np.random.choice([0,1], N, p=[0.7,0.3])
stop         = np.random.choice([0,1], N, p=[0.8,0.2])
amenity      = np.random.choice([0,1], N, p=[0.7,0.3])
day_of_week  = np.random.choice(
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
    N, p=[0.16,0.15,0.15,0.15,0.17,0.12,0.10]
)
month  = np.random.choice(range(1,13), N)
location  = np.random.choice(locations, N)

severity_score = (
    (speed_limit / 120) * 2.5 +
    (wind_speed / 50) * 1.5 +
    ((15 - visibility) / 15) * 1.5 +
    np.isin(weather, ['Heavy Rain','Fog','Thunderstorm']).astype(float) * 1.5 +
    ((temperature > 35)).astype(float) * 0.5 +
    (hours < 6).astype(float) * 0.8 +
    np.random.normal(0, 0.5, N)
)

severity_bins = np.percentile(severity_score, [40, 65, 85])
severity      = np.clip(np.digitize(severity_score, bins=severity_bins) + 1, 1, 4)

df = pd.DataFrame({
    'Severity':         severity,
    'Temperature_C':    np.round(temperature, 1),
    'Wind_Speed_kmh':   np.round(wind_speed, 1),
    'Visibility_km':    np.round(visibility, 1),
    'Precipitation_mm': np.round(precipitation, 2),
    'Humidity_pct':     np.round(humidity, 1),
    'Pressure_hPa':     np.round(pressure, 2),
    'Speed_Limit':      speed_limit,
    'Weather_Condition':weather,
    'Road_Type':        road_type,
    'Hour':             hours,
    'Day_of_Week':      day_of_week,
    'Month':            month,
    'State':            location, 
    'Junction':         junction,
    'Traffic_Signal':   traffic_signal,
    'Crossing':         crossing,
    'Stop':             stop,
    'Amenity':          amenity,
    'Country':          'India'
})

os.makedirs('data', exist_ok=True)
df.to_csv('data/accidents_cleaned.csv', index=False)
print("Dataset created: data/accidents_cleaned.csv")
print(df['Severity'].value_counts().sort_index())