"""
FinSentinel Dashboard - Real-time trading signals from social media sentiment
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import time
import sqlite3
import asyncio

from utils.database import db
from orchestrator import orchestrator
from config import DASHBOARD_REFRESH_INTERVAL, FINANCIAL_INFLUENCERS

# Page configuration
st.set_page_config(
    page_title="FinSentinel - AI Trading Signals",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .signal-buy {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .signal-sell {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .signal-hold {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class FinSentinelDashboard:
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize Streamlit session state"""
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = DASHBOARD_REFRESH_INTERVAL
    
    def load_recent_signals(self, limit: int = 50) -> pd.DataFrame:
        """Load recent trading signals from database"""
        try:
            df = db.get_recent_signals(limit)
            if df.empty:
                return pd.DataFrame()
            
            # Convert timestamp columns
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['generated_at'] = pd.to_datetime(df['generated_at'])
            
            # Add time ago column
            df['time_ago'] = df['generated_at'].apply(
                lambda x: self.format_time_ago(datetime.now() - x)
            )
            
            return df
            
        except Exception as e:
            st.error(f"Error loading signals: {e}")
            return pd.DataFrame()
    
    def format_time_ago(self, delta: timedelta) -> str:
        """Format timedelta as human-readable string"""
        total_seconds = int(delta.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds}s ago"
        elif total_seconds < 3600:
            return f"{total_seconds // 60}m ago"
        elif total_seconds < 86400:
            return f"{total_seconds // 3600}h ago"
        else:
            return f"{total_seconds // 86400}d ago"
    
    def get_system_metrics(self) -> dict:
        """Get system performance metrics"""
        try:
            status = orchestrator.get_system_status()
            
            # Get signal counts from database
            with sqlite3.connect(db.db_path) as conn:
                cursor = conn.cursor()
                
                # Total signals today
                cursor.execute("""
                    SELECT COUNT(*) FROM trading_signals 
                    WHERE DATE(generated_at) = DATE('now')
                """)
                signals_today = cursor.fetchone()[0]
                
                # Signal distribution
                cursor.execute("""
                    SELECT signal, COUNT(*) FROM trading_signals 
                    WHERE generated_at >= datetime('now', '-24 hours')
                    GROUP BY signal
                """)
                signal_dist = dict(cursor.fetchall())
                
                # Top assets
                cursor.execute("""
                    SELECT asset_symbol, COUNT(*) as count FROM trading_signals 
                    WHERE generated_at >= datetime('now', '-24 hours')
                    GROUP BY asset_symbol 
                    ORDER BY count DESC LIMIT 5
                """)
                top_assets = cursor.fetchall()
            
            return {
                'is_running': status['is_running'],
                'signals_today': signals_today,
                'total_processed': status['stats']['tweets_processed'],
                'errors': status['stats']['errors'],
                'signal_distribution': signal_dist,
                'top_assets': top_assets,
                'last_run': status['stats']['last_run']
            }
            
        except Exception as e:
            st.error(f"Error getting metrics: {e}")
            return {}
    
    def render_header(self):
        """Render dashboard header"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.title("📈 FinSentinel")
            st.caption("AI-Powered Social Media Trading Signals")
        
        with col2:
            if st.button("🔄 Refresh Data"):
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        with col3:
            st.session_state.auto_refresh = st.checkbox(
                "Auto Refresh", 
                value=st.session_state.auto_refresh
            )
    
    def render_metrics(self, metrics: dict):
        """Render key metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Signals Today",
                value=metrics.get('signals_today', 0),
                delta=None
            )
        
        with col2:
            st.metric(
                "📊 Total Processed",
                value=metrics.get('total_processed', 0),
                delta=None
            )
        
        with col3:
            st.metric(
                "⚠️ Errors",
                value=metrics.get('errors', 0),
                delta=None
            )
        
        with col4:
            status = "🟢 Running" if metrics.get('is_running', False) else "🔴 Stopped"
            st.metric(
                "System Status",
                value=status,
                delta=None
            )
    
    def render_signal_charts(self, df: pd.DataFrame):
        """Render signal analysis charts"""
        if df.empty:
            st.warning("No signals data available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Signal distribution pie chart
            signal_counts = df['signal'].value_counts()
            
            fig_pie = px.pie(
                values=signal_counts.values,
                names=signal_counts.index,
                title="Signal Distribution (Last 50)",
                color_discrete_map={
                    'BUY': '#28a745',
                    'SELL': '#dc3545', 
                    'HOLD': '#ffc107'
                }
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Signal strength histogram
            fig_hist = px.histogram(
                df,
                x='signal_strength',
                nbins=20,
                title="Signal Strength Distribution",
                labels={'signal_strength': 'Signal Strength', 'count': 'Count'}
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
    
    def render_asset_analysis(self, df: pd.DataFrame):
        """Render asset-specific analysis"""
        if df.empty:
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top assets by signal count
            asset_counts = df['asset_symbol'].value_counts().head(10)
            
            fig_bar = px.bar(
                x=asset_counts.index,
                y=asset_counts.values,
                title="Most Mentioned Assets",
                labels={'x': 'Asset Symbol', 'y': 'Signal Count'}
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Sentiment vs signal strength scatter
            fig_scatter = px.scatter(
                df,
                x='sentiment_score',
                y='signal_strength',
                color='signal',
                size='signal_strength',
                hover_data=['asset_symbol', 'influencer'],
                title="Sentiment vs Signal Strength",
                color_discrete_map={
                    'BUY': '#28a745',
                    'SELL': '#dc3545',
                    'HOLD': '#ffc107'
                }
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    def render_signals_table(self, df: pd.DataFrame):
        """Render the main signals table"""
        if df.empty:
            st.warning("No recent signals found")
            return
        
        st.subheader("📋 Recent Trading Signals")
        
        # Create a formatted dataframe for display
        display_df = df.copy()
        
        # Format columns
        display_df['Signal Strength'] = display_df['signal_strength'].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
        )
        
        display_df['Sentiment Score'] = display_df['sentiment_score'].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
        )
        
        # Select and rename columns for display
        columns_to_show = {
            'time_ago': 'Time',
            'influencer': 'Influencer',
            'content': 'Tweet Content',
            'asset_symbol': 'Asset',
            'sentiment_label': 'Sentiment',
            'signal': 'Signal',
            'Signal Strength': 'Strength',
            'reasoning': 'AI Reasoning'
        }
        
        display_df = display_df[list(columns_to_show.keys())].rename(columns=columns_to_show)
        
        # Truncate tweet content for display
        display_df['Tweet Content'] = display_df['Tweet Content'].apply(
            lambda x: x[:100] + "..." if len(str(x)) > 100 else str(x)
        )
        
        display_df['AI Reasoning'] = display_df['AI Reasoning'].apply(
            lambda x: x[:150] + "..." if len(str(x)) > 150 else str(x)
        )
        
        # Style the dataframe
        def style_signal(val):
            if val == 'BUY':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'SELL':
                return 'background-color: #f8d7da; color: #721c24'
            elif val == 'HOLD':
                return 'background-color: #fff3cd; color: #856404'
            return ''
        
        styled_df = display_df.style.applymap(style_signal, subset=['Signal'])
        
        st.dataframe(styled_df, use_container_width=True, height=400)
    
    def render_influencer_performance(self):
        """Render influencer performance metrics"""
        try:
            perf_df = db.get_influencer_performance()
            
            if not perf_df.empty:
                st.subheader("👥 Influencer Performance")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Signal count by influencer
                    fig_inf = px.bar(
                        perf_df.head(10),
                        x='influencer',
                        y='total_signals',
                        title="Signals by Influencer"
                    )
                    fig_inf.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_inf, use_container_width=True)
                
                with col2:
                    # High confidence rate
                    fig_conf = px.bar(
                        perf_df.head(10),
                        x='influencer',
                        y='high_confidence_rate',
                        title="High Confidence Signal Rate"
                    )
                    fig_conf.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_conf, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error loading influencer performance: {e}")
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🔧 Controls")
        
        # System controls
        if st.sidebar.button("▶️ Run Single Cycle"):
            with st.spinner("Running monitoring cycle..."):
                stats = orchestrator.run_single_cycle()
                st.sidebar.success(f"Cycle completed: {stats['tweets_processed']} tweets processed")
        
        # Filters
        st.sidebar.header("🔍 Filters")
        
        # Signal type filter
        signal_filter = st.sidebar.multiselect(
            "Signal Types",
            options=['BUY', 'SELL', 'HOLD'],
            default=['BUY', 'SELL', 'HOLD']
        )
        
        # Asset type filter
        asset_filter = st.sidebar.multiselect(
            "Asset Types",
            options=['stock', 'crypto'],
            default=['stock', 'crypto']
        )
        
        # Time range
        time_range = st.sidebar.selectbox(
            "Time Range",
            options=['Last 1 hour', 'Last 6 hours', 'Last 24 hours', 'Last 7 days'],
            index=2
        )
        
        # Influencer filter
        influencer_filter = st.sidebar.multiselect(
            "Influencers",
            options=FINANCIAL_INFLUENCERS,
            default=FINANCIAL_INFLUENCERS[:5]  # Default to first 5
        )
        
        return {
            'signals': signal_filter,
            'assets': asset_filter,
            'time_range': time_range,
            'influencers': influencer_filter
        }
    
    def apply_filters(self, df: pd.DataFrame, filters: dict) -> pd.DataFrame:
        """Apply filters to the dataframe"""
        if df.empty:
            return df
        
        filtered_df = df.copy()
        
        # Signal filter
        if filters['signals']:
            filtered_df = filtered_df[filtered_df['signal'].isin(filters['signals'])]
        
        # Influencer filter
        if filters['influencers']:
            filtered_df = filtered_df[filtered_df['influencer'].isin(filters['influencers'])]
        
        # Time range filter
        now = datetime.now()
        time_mapping = {
            'Last 1 hour': now - timedelta(hours=1),
            'Last 6 hours': now - timedelta(hours=6),
            'Last 24 hours': now - timedelta(hours=24),
            'Last 7 days': now - timedelta(days=7)
        }
        
        if filters['time_range'] in time_mapping:
            cutoff = time_mapping[filters['time_range']]
            filtered_df = filtered_df[filtered_df['generated_at'] >= cutoff]
        
        return filtered_df
    
    def run(self):
        """Main dashboard execution"""
        # Auto-refresh logic
        if st.session_state.auto_refresh:
            time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
            if time_since_refresh >= st.session_state.refresh_interval:
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        # Render components
        self.render_header()
        
        # Load data
        with st.spinner("Loading data..."):
            signals_df = self.load_recent_signals(100)
            metrics = self.get_system_metrics()
        
        # Render sidebar and get filters
        filters = self.render_sidebar()
        
        # Apply filters
        filtered_df = self.apply_filters(signals_df, filters)
        
        # Render main content
        self.render_metrics(metrics)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_signal_charts(filtered_df)
        
        with col2:
            self.render_asset_analysis(filtered_df)
        
        st.markdown("---")
        
        # Signals table
        self.render_signals_table(filtered_df)
        
        st.markdown("---")
        
        # Influencer performance
        self.render_influencer_performance()
        
        # Footer
        st.markdown("---")
        st.caption(f"Last updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Auto-refresh placeholder
        if st.session_state.auto_refresh:
            time.sleep(1)
            st.rerun()

# Main execution
if __name__ == "__main__":
    dashboard = FinSentinelDashboard()
    dashboard.run()
