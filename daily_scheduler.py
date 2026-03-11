"""
DAILY SCHEDULER
===============
Runs auto-update system every day at 1:00 AM
"""

import schedule
import time
from datetime import datetime
from auto_update_system import AutoUpdateSystem

def scheduled_update_job():
    """Job that runs at scheduled time"""
    print(f"\n{'='*60}")
    print(f"⏰ SCHEDULED UPDATE TRIGGERED")
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Run update
    updater = AutoUpdateSystem()
    success = updater.run_daily_update()
    
    if success:
        print(f"\n✅ Scheduled update completed successfully!")
    else:
        print(f"\n❌ Scheduled update failed!")
    
    print(f"\n{'='*60}")
    print(f"⏰ Next update scheduled for tomorrow at 01:00 AM")
    print(f"{'='*60}\n")


# Schedule to run daily at 1:00 AM
schedule.every().day.at("01:00").do(scheduled_update_job)

# For testing: Run every 5 minutes
# schedule.every(5).minutes.do(scheduled_update_job)

print("""
╔════════════════════════════════════════════════╗
║   PM2.5 DAILY SCHEDULER - RUNNING             ║
║                                                ║
║   Schedule: Every day at 01:00 AM              ║
║   Status:   Waiting for next scheduled time    ║
║                                                ║
║   Press Ctrl+C to stop                         ║
╚════════════════════════════════════════════════╝
""")

print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Next run: Tomorrow at 01:00 AM\n")

# Keep running
try:
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute
        
except KeyboardInterrupt:
    print("\n\n⏹️  Scheduler stopped by user.")