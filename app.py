import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# Page Config & High-End Aesthetic Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="ReadyNest | Analytics Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for Glassmorphism & Neon Dark Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0B0F19;
    }
    .stMetric {
        background: #131927;
        border: 1px solid #1F293D;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .stMetric label {
        color: #64748B !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 0.8px;
    }
    .stMetric div[data-testid="stMetricValue"] {
        color: #00F2FE !important;
        font-weight: 700 !important;
        font-size: 26px !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Synthetic Dataset Generator
# ---------------------------------------------------------
@st.cache_data
def load_messenger_data():
    np.random.seed(42)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    dates = pd.date_range(start=start_date, end=end_date, freq='h')
    
    users = ["Sneha Verma", "Rahul Sharma", "Aman Singh", "Pooja (Design Team)", "Mehak Gupta", "John Doe", "Amar (Project Team)", "User_102"]
    chat_types = ["1-to-1", "Group", "Broadcast"]
    
    data = []
    for d in dates:
        num_msgs = np.random.poisson(lam=12 if 9 <= d.hour <= 22 else 3)
        for _ in range(num_msgs):
            user = np.random.choice(users, p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.05, 0.05, 0.05])
            c_type = np.random.choice(chat_types, p=[0.6, 0.35, 0.05])
            data.append({
                "timestamp": d,
                "date": d.date(),
                "hour": d.hour,
                "user": user,
                "chat_type": c_type,
                "has_attachment": np.random.choice([0, 1], p=[0.85, 0.15])
            })
            
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df

df_raw = load_messenger_data()

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.markdown("<h2 style='color:#00F2FE;'>⚡ ReadyNest</h2>", unsafe_allow_html=True)
st.sidebar.caption("ANALYTICS DASHBOARD CONTROL")

min_date = df_raw['date'].min().to_pydatetime()
max_date = df_raw['date'].max().to_pydatetime()

date_range = st.sidebar.date_input(
    "Date Range Filter",
    value=[min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

selected_chat_type = st.sidebar.multiselect(
    "Chat Channels",
    options=df_raw['chat_type'].unique(),
    default=df_raw['chat_type'].unique()
)

# Apply Filters
if len(date_range) == 2:
    start_filter, end_filter = date_range[0], date_range[1]
    filtered_df = df_raw[
        (df_raw['date'] >= pd.to_datetime(start_filter)) & 
        (df_raw['date'] <= pd.to_datetime(end_filter)) &
        (df_raw['chat_type'].isin(selected_chat_type))
    ]
else:
    filtered_df = df_raw

# ---------------------------------------------------------
# Header & Key Metrics
# ---------------------------------------------------------
st.markdown("<h1 style='color:#FFFFFF; margin-bottom:0px;'>ReadyNest Messenger</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#00F2FE; font-weight:600; font-size:12px; letter-spacing:1px; margin-top:0px;'>REAL-TIME ENGAGEMENT & PREDICTIVE FORECASTING DASHBOARD</p>", unsafe_allow_html=True)

daily_users = filtered_df.groupby(filtered_df['timestamp'].dt.date)['user'].nunique()
avg_dau = int(daily_users.mean()) if not daily_users.empty else 0
total_messages = len(filtered_df)
active_users = filtered_df['user'].nunique()
media_shared = filtered_df['has_attachment'].sum()

m1, m2, m3, m4 = st.columns(4)
m1.metric("DAILY ACTIVE USERS (DAU)", f"{avg_dau:,}", "+12% vs last week")
m2.metric("TOTAL MESSAGES SENT", f"{total_messages:,}", "Across active channels")
m3.metric("ACTIVE COMMUNICATORS", f"{active_users} Users", "100% engagement")
m4.metric("MEDIA & ATTACHMENTS", f"{media_shared:,}", "15% of total volume")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Aesthetic Charts Section
# ---------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.markdown("<h4 style='color:#FFFFFF;'>Message Volume Trend</h4>", unsafe_allow_html=True)
    daily_msg = filtered_df.groupby('date').size().reset_index(name='count')
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=daily_msg['date'], y=daily_msg['count'],
        mode='lines+markers',
        fill='tozeroy',
        fillcolor='rgba(0, 242, 254, 0.1)',
        line=dict(color='#00F2FE', width=2.5),
        marker=dict(size=5, color='#00F2FE')
    ))
    fig1.update_layout(
        template="plotly_dark",
        paper_bgcolor='#131927',
        plot_bgcolor='#131927',
        margin=dict(l=20, r=20, t=20, b=20),
        height=320,
        xaxis=dict(gridcolor='#1F293D'),
        yaxis=dict(gridcolor='#1F293D')
    )
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    st.markdown("<h4 style='color:#FFFFFF;'>Top Communicators</h4>", unsafe_allow_html=True)
    user_counts = filtered_df['user'].value_counts().head(5).reset_index()
    user_counts.columns = ['user', 'count']
    
    fig2 = px.bar(
        user_counts, x='count', y='user', orientation='h',
        color='count',
        color_continuous_scale=['#FF007F', '#7F00FF', '#00F2FE']
    )
    fig2.update_layout(
        template="plotly_dark",
        paper_bgcolor='#131927',
        plot_bgcolor='#131927',
        margin=dict(l=20, r=20, t=20, b=20),
        height=320,
        yaxis=dict(autorange="reversed", gridcolor='#1F293D'),
        xaxis=dict(gridcolor='#1F293D'),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------
# BONUS FEATURE: Predictive Engagement Trends ML Chart
# ---------------------------------------------------------
st.markdown("<h4 style='color:#FFFFFF;'>Predictive Engagement Trends (ML Regression Forecast)</h4>", unsafe_allow_html=True)

daily_agg = filtered_df.groupby('date').size().reset_index(name='y')
daily_agg['x'] = (daily_agg['date'] - daily_agg['date'].min()).dt.days

if len(daily_agg) > 5:
    X = daily_agg[['x']].values
    y = daily_agg['y'].values
    
    model = LinearRegression().fit(X, y)
    
    future_days = 14
    last_x = daily_agg['x'].max()
    future_x = np.array([last_x + i for i in range(1, future_days + 1)]).reshape(-1, 1)
    future_y = model.predict(future_x)
    
    last_date = daily_agg['date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
    
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=daily_agg['date'], y=daily_agg['y'],
        mode='lines', name='Historical Activity',
        line=dict(color='#00F2FE', width=2)
    ))
    fig_pred.add_trace(go.Scatter(
        x=future_dates, y=future_y,
        mode='lines+markers', name='14-Day ML Forecast (Bonus Feature)',
        line=dict(color='#FF007F', width=2.5, dash='dash'),
        marker=dict(symbol='square', size=6)
    ))
    
    fig_pred.update_layout(
        template="plotly_dark",
        paper_bgcolor='#131927',
        plot_bgcolor='#131927',
        margin=dict(l=20, r=20, t=20, b=20),
        height=350,
        xaxis=dict(gridcolor='#1F293D'),
        yaxis=dict(gridcolor='#1F293D'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_pred, use_container_width=True)