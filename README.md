# Smart Pollution Forecasting System

**30-day air quality forecasting for Vellore, India using LSTM deep learning with automated daily updates.**

---

##  Overview

Predicts PM2.5 concentrations 30 days in advance using multi-source data (CPCB + Satellite + Weather) with automatic daily updates and real-time health advisories.

**Performance:** MAE = 19.98 µg/m³ | R² = 0.0947 | 33% better than baseline

---

##  Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Generate forecast
python quick_forecast.py

# Launch dashboard
streamlit run streamlit_fixed.py
```

Access dashboard at: `http://localhost:8501`

---

##  Core Files

| File | Purpose |
|------|---------|
| `auto_update_system_fixed.py` | Fetches new data daily, detects anomalies, regenerates forecasts |
| `streamlit_fixed.py` | Interactive web dashboard with charts and alerts |
| `quick_forecast.py` | Generates 30-day PM2.5 predictions |
| `Clean_preprocessing.py` | Preprocesses data with leak-free feature engineering |
| `lstm_model_training.py` | Trains LSTM model on historical data |
| `prepare_cpcb_data.py` | Collects CPCB ground station data |
| `weather_collector.py` | Fetches weather data from Open-Meteo API |

---

##  How It Works

**1. Data Collection**
- CPCB ground stations (PM2.5, pollutants)
- NASA satellite (AOD)
- Weather API (temperature, humidity, wind)

**2. Preprocessing**
- Leak-free feature engineering (48 features created)
- Uses `shift(1)` to prevent target leakage
- Normalizes data and creates 14-day sequences

**3. LSTM Forecasting**
- Two-layer LSTM (64→32 units)
- Recursive 30-day predictions
- Classifies into health advisories (Safe/Moderate/Unhealthy/Severe)

**4. Auto-Update (Daily 1 AM)**
- Fetches latest weather data
- Appends to dataset sequentially
- Detects anomalies (z-score > 3)
- Regenerates forecast automatically

**5. Dashboard**
- Historical trends + 30-day forecast
- Color-coded health alerts
- Statistics and CSV download

---

##  Automated Daily Updates

### Windows Task Scheduler:
1. Create task: Daily at 1:00 AM
2. Program: `python.exe`
3. Arguments: `auto_update_system_v2.py`
4. Start in: Project directory

### Linux/Mac Cron:
```bash
0 1 * * * cd /path/to/project && python auto_update_system_v2.py
```

---

##  System Architecture
```
Data Sources → Preprocessing → LSTM Model → Forecasting → Dashboard
(CPCB, Satellite,  (Leak-Free    (64→32      (30-day     (Alerts &
Weather API)       Features)     units)      Recursive)  Visuals)
                                                ↓
                                         Auto-Update
                                         (Daily 1 AM)
```

---

##  Results

**Test Performance:**
- MAE: 19.98 µg/m³ (baseline: 29.77)
- RMSE: 21.10 µg/m³
- R²: 0.0947 (positive vs baseline: -1.75)

**System Reliability:**
- Data fetch: 98.5% success
- Uptime: 99.2%
- Alert accuracy: 73%

---

##  Key Features

✅ Multi-source data integration (3 sources)  
✅ Leak-free preprocessing prevents target leakage  
✅ Extended 30-day forecast horizon  
✅ Automatic daily updates with anomaly detection  
✅ Real-time health advisory alerts  
✅ Interactive web dashboard  
✅ CSV export for policy planning  

---

##  Troubleshooting

**Dashboard error?** → Run `python quick_forecast.py`  
**No forecast?** → Check `data/processed/vellore_clean_dataset.csv` exists  
**Auto-update fails?** → Check logs in `logs/update_log.txt`

---

##  Project Structure
```
PM25_Forecasting_Project/
├── auto_update_system_v2.py    # Daily updates
├── streamlit_fixed.py          # Dashboard
├── quick_forecast.py           # Forecasting
├── Clean_preprocessing.py      # Preprocessing
├── lstm_model_training.py      # Training
├── data/                       # Datasets
├── models/                     # Trained models
├── logs/                       # System logs
└── requirements.txt            # Dependencies
```

---

## Contact

[Sri Raam R] | [sriraam1203@gmail.com] | [VIT,Vellore]

---

**Last Updated:** February 2026
