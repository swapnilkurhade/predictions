pip install fastapi uvicorn
uvicorn main:app --reload 

POST api = /schedule

input request format : 
{
  "planting_date": "2025-06-01",
  "plantation_type": "Setling (nursery plant)",
  "seedling_weeks": 5,
  "cane_variety": "Co 86032",
  "soil": {
    "Ph": 6.5,
    "Electrical_Conductivity": 0.3,
    "Organic_Carbon": 0.8,
    "Available_Nitrogen": 250,
    "Available_Phosphorus": 18,
    "Available_Potassium": 320,
    "Exchangeable_Sodium": 0.2,
    "Free_Lime": 2.0,
    "Iron": 4.2,
    "Manganese": 3.1,
    "Zinc": 0.8,
    "Copper": 0.4,
    "Sulphar": 10,
    "Boron": 0.5
  },
  "basal_n_pct": 30,
  "basal_p_pct": 40,
  "basal_k_pct": 20,
  "include_weather": true,
  "weather": {
    "temperature": 33,
    "rainfall": 12,
    "evapotranspiration": 5.5,
    "humidity": 60,
    "wind_speed": 8,
    "ndvi": 0.35,
    "evi": 0.22,
    "soil_moisture": 0.18,
    "forecast_temp_avg": 31,
    "forecast_rainfall_total": 70,
    "lux_intensity": 25000,
    "ph": 6.5
  },
  "adjustment": {
    "mode": "current_and_next"
  }
}


 
