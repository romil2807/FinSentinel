"""
FinSentinel - Main Application Entry Point
"""
import argparse
import asyncio
import logging
import sys
from datetime import datetime

from orchestrator import orchestrator
from utils.alerts import alert_manager
from utils.database import db

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_database():
    """Initialize the database"""
    try:
        db.init_database()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

def run_continuous():
    """Run FinSentinel continuously"""
    logger.info("🚀 Starting FinSentinel in continuous mode...")
    
    if not setup_database():
        sys.exit(1)
    
    try:
        asyncio.run(orchestrator.run_continuous())
    except KeyboardInterrupt:
        logger.info("FinSentinel stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

def run_single_cycle():
    """Run a single monitoring cycle"""
    logger.info("🔄 Running single FinSentinel cycle...")
    
    if not setup_database():
        sys.exit(1)
    
    try:
        stats = orchestrator.run_single_cycle()
        
        print("\n" + "="*50)
        print("📊 FINSENTINEL CYCLE RESULTS")
        print("="*50)
        print(f"📈 Tweets Processed: {stats['tweets_processed']}")
        print(f"🎯 Signals Generated: {stats['signals_generated']}")
        print(f"⚠️  Errors: {stats['errors']}")
        print(f"🕐 Last Run: {stats['last_run']}")
        print("="*50)
        
        return stats
        
    except Exception as e:
        logger.error(f"Error in single cycle: {e}")
        sys.exit(1)

def show_status():
    """Show system status"""
    logger.info("📋 Checking FinSentinel status...")
    
    try:
        status = orchestrator.get_system_status()
        
        print("\n" + "="*50)
        print("🖥️  FINSENTINEL SYSTEM STATUS")
        print("="*50)
        print(f"System Running: {'✅ Yes' if status['is_running'] else '❌ No'}")
        print(f"Last Processing: {status['last_processing_time'] or 'Never'}")
        print(f"Tweets Processed: {status['stats']['tweets_processed']}")
        print(f"Signals Generated: {status['stats']['signals_generated']}")
        print(f"Errors: {status['stats']['errors']}")
        print("\nComponent Status:")
        for component, active in status['components'].items():
            print(f"  {component}: {'✅' if active else '❌'}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error checking status: {e}")
        sys.exit(1)

def test_alerts():
    """Test the alert system"""
    logger.info("🧪 Testing alert system...")
    
    try:
        results = alert_manager.test_alerts()
        
        print("\n" + "="*40)
        print("🔔 ALERT SYSTEM TEST RESULTS")
        print("="*40)
        print(f"Slack: {'✅ Success' if results.get('slack', False) else '❌ Failed'}")
        print(f"Discord: {'✅ Success' if results.get('discord', False) else '❌ Failed'}")
        print(f"Overall: {'✅ Success' if results.get('sent', False) else '❌ Failed'}")
        print("="*40)
        
    except Exception as e:
        logger.error(f"Error testing alerts: {e}")
        sys.exit(1)

def run_dashboard():
    """Run the Streamlit dashboard"""
    import subprocess
    import os
    
    logger.info("🖥️  Starting FinSentinel dashboard...")
    
    try:
        # Get the dashboard file path
        dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "streamlit_app.py")
        
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            dashboard_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        sys.exit(1)

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="FinSentinel - AI-Powered Social Media Trading Signals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run continuously
  python main.py --mode single      # Run single cycle
  python main.py --mode dashboard   # Run dashboard
  python main.py --mode status      # Show status
  python main.py --mode test-alerts # Test alerts
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['continuous', 'single', 'dashboard', 'status', 'test-alerts'],
        default='continuous',
        help='Operation mode (default: continuous)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Print banner
    print("""
    ╔═══════════════════════════════════════╗
    ║         🔍 FinSentinel v1.0           ║
    ║   AI-Powered Trading Signal Engine    ║
    ║     Real-time Social Sentiment        ║
    ╚═══════════════════════════════════════╝
    """)
    
    # Route to appropriate function
    if args.mode == 'continuous':
        run_continuous()
    elif args.mode == 'single':
        run_single_cycle()
    elif args.mode == 'dashboard':
        run_dashboard()
    elif args.mode == 'status':
        show_status()
    elif args.mode == 'test-alerts':
        test_alerts()

if __name__ == "__main__":
    main()
