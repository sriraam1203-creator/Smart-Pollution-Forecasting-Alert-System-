"""
MANUAL TEST - Auto Update System
=================================
Run this to test the auto-update system manually
"""

from auto_update_system_fixed import AutoUpdateSystem
from datetime import datetime

print("""
╔════════════════════════════════════════════╗
║      MANUAL TEST - AUTO UPDATE SYSTEM      ║
╚════════════════════════════════════════════╝

This will simulate a daily update:
1. Fetch latest weather data
2. Estimate PM2.5 (or fetch from API if available)
3. Append to dataset
4. Check for anomalies
5. Regenerate forecast
6. Save alerts if needed

Press Enter to continue...
""")

input()

# Create updater
updater = AutoUpdateSystem()

# Run update
print("\n" + "="*60)
print("STARTING MANUAL UPDATE TEST")
print("="*60 + "\n")

success = updater.run_daily_update()

if success:
    print("""
    
╔════════════════════════════════════════════╗
║           ✅ TEST SUCCESSFUL!               ║
╚════════════════════════════════════════════╝

What happened:
✓ Latest data fetched
✓ Dataset updated
✓ Anomaly check completed
✓ Forecast regenerated
✓ Logs saved

Check these files:
📁 data/processed/vellore_clean_dataset.csv (updated dataset)
📁 data/outputs/vellore_30day_forecast.csv (new forecast)
📁 logs/update_log.txt (detailed logs)
📁 logs/alerts.json (any anomaly alerts)

Next steps:
1. Run streamlit dashboard to see updated forecast
2. Set up daily_scheduler.py to automate
3. Deploy to server for continuous operation
    """)
else:
    print("""
    
╔════════════════════════════════════════════╗
║           ❌ TEST FAILED                    ║
╚════════════════════════════════════════════╝

Check logs/update_log.txt for error details
    """)