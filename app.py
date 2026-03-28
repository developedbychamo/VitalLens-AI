"""
VitalLens - Health Scanner Using Just Your Webcam
Advanced Health Monitoring: Heart Rate, Stress, HRV & Breathing Rate
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import time
import threading
from collections import deque
from scipy import signal
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="VitalLens - AI Health Scanner",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional design
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        color: #a0a0a0;
        font-size: 1rem;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.95);
        border-radius: 1rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
        border-top: 3px solid #667eea;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-trend {
        font-size: 0.7rem;
        margin-top: 0.25rem;
    }
    
    .status-card {
        background: rgba(255,255,255,0.95);
        border-radius: 1rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #10b981;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        border-radius: 0.5rem;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
    
    .timer {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        color: #667eea;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff;
    }
    
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'scanning' not in st.session_state:
    st.session_state.scanning = False
if 'scan_complete' not in st.session_state:
    st.session_state.scan_complete = False
if 'rgb_history' not in st.session_state:
    st.session_state.rgb_history = deque(maxlen=900)
if 'timestamp_history' not in st.session_state:
    st.session_state.timestamp_history = deque(maxlen=900)
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = {}
if 'hr_series' not in st.session_state:
    st.session_state.hr_series = None

# Helper functions
def extract_rgb_signal(frame):
    """Extract average RGB values from face region"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        
        r_mean = np.mean(face_roi[:,:,2])
        g_mean = np.mean(face_roi[:,:,1])
        b_mean = np.mean(face_roi[:,:,0])
        
        return np.array([r_mean, g_mean, b_mean]), faces[0]
    
    return None, None

def calculate_heart_rate(rgb_signal, timestamps, fps=30):
    """Calculate heart rate from RGB signal"""
    if len(rgb_signal) < 100:
        return None, None
    
    green_signal = rgb_signal[:, 1]
    green_signal = signal.detrend(green_signal)
    green_signal = (green_signal - np.mean(green_signal)) / np.std(green_signal)
    
    nyquist = fps / 2
    low = 0.8 / nyquist
    high = 3.0 / nyquist
    
    try:
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, green_signal)
    except:
        return None, None
    
    peaks, properties = find_peaks(filtered_signal, distance=fps*0.5, prominence=0.5)
    
    if len(peaks) < 2:
        return None, None
    
    peak_intervals = np.diff(peaks) / fps
    heart_rates = 60 / peak_intervals
    valid_hr = heart_rates[(heart_rates >= 40) & (heart_rates <= 180)]
    
    if len(valid_hr) > 0:
        return np.mean(valid_hr), valid_hr
    
    return None, None

def calculate_hrv(heart_rates):
    """Calculate Heart Rate Variability"""
    if heart_rates is None or len(heart_rates) < 5:
        return None
    
    rr_intervals = 60 / np.array(heart_rates)
    successive_diffs = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(successive_diffs**2))
    sdnn = np.std(rr_intervals)
    
    return {
        'rmssd': rmssd,
        'sdnn': sdnn,
        'status': 'Normal' if 20 < rmssd < 60 else 'Elevated' if rmssd > 60 else 'Reduced'
    }

def calculate_breathing_rate(rgb_signal, fps=30):
    """Calculate breathing rate from RGB signal"""
    if len(rgb_signal) < 100:
        return None
    
    red_signal = rgb_signal[:, 0]
    blue_signal = rgb_signal[:, 2]
    ratio_signal = red_signal / (blue_signal + 1e-6)
    ratio_signal = signal.detrend(ratio_signal)
    ratio_signal = (ratio_signal - np.mean(ratio_signal)) / np.std(ratio_signal)
    
    nyquist = fps / 2
    low = 0.1 / nyquist
    high = 0.5 / nyquist
    
    try:
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, ratio_signal)
    except:
        return None
    
    peaks, _ = find_peaks(filtered_signal, distance=fps*2, prominence=0.3)
    
    if len(peaks) < 2:
        return None
    
    peak_intervals = np.diff(peaks) / fps
    breathing_rates = 60 / peak_intervals
    valid_br = breathing_rates[(breathing_rates >= 6) & (breathing_rates <= 30)]
    
    if len(valid_br) > 0:
        return np.mean(valid_br)
    
    return None

def calculate_stress_level(heart_rate, hrv, breathing_rate):
    """Calculate stress level based on multiple metrics"""
    stress_score = 0
    
    if heart_rate:
        if heart_rate > 100:
            stress_score += 40
        elif heart_rate > 80:
            stress_score += 20
    
    if hrv and isinstance(hrv, dict) and hrv.get('rmssd'):
        if hrv['rmssd'] < 20:
            stress_score += 40
        elif hrv['rmssd'] < 35:
            stress_score += 20
    
    if breathing_rate:
        if breathing_rate > 20:
            stress_score += 20
        elif breathing_rate > 16:
            stress_score += 10
    
    if stress_score >= 70:
        return "High Stress", stress_score, "#ef4444"
    elif stress_score >= 40:
        return "Moderate Stress", stress_score, "#f59e0b"
    else:
        return "Low Stress", stress_score, "#10b981"

# Header
st.markdown("""
<div class="hero">
    <div class="hero-title">✨ VitalLens</div>
    <div class="hero-subtitle">Health Scanner Using Just Your Webcam</div>
    <div class="hero-subtitle" style="font-size: 0.85rem;">Point your camera at your face for 30 seconds | Get heart rate, stress level, HRV & breathing rate</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("###  About VitalLens")
    st.markdown("""
    **No wearable, no hardware** — just AI + your camera.
    
    **What we measure:**
    -  Heart Rate (BPM)
    -  Heart Rate Variability (HRV)
    -  Breathing Rate (breaths/min)
    -  Stress Level
    """)
    
    st.markdown("---")
    st.markdown("###  How It Works")
    st.markdown("""
    1. Position your face in frame
    2. Click 'Start Scan'
    3. Stay still for 30 seconds
    4. Get your complete health report
    """)
    
    st.markdown("---")
    st.markdown("###  Tips for Best Results")
    st.markdown("""
    - Good lighting (natural light works best)
    - Face directly facing camera
    - Remove glasses if possible
    - Stay still during scan
    - Distance: 1-2 feet from camera
    """)
    
    st.markdown("---")
    st.markdown("###  Health Ranges")
    st.markdown("""
    | Metric | Normal Range |
    |--------|-------------|
    | Heart Rate | 60-100 BPM |
    | HRV (RMSSD) | 20-60 ms |
    | Breathing | 12-20 breaths/min |
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("###  Camera Feed")
    camera_placeholder = st.empty()
    
    col_start, col_stop = st.columns(2)
    with col_start:
        start_scan = st.button(" Start 30-Second Scan", use_container_width=True)
    with col_stop:
        stop_scan = st.button(" Stop Scan", use_container_width=True)
    
    if start_scan:
        st.session_state.scanning = True
        st.session_state.scan_complete = False
        st.session_state.rgb_history.clear()
        st.session_state.timestamp_history.clear()
        st.session_state.scan_results = {}
        st.session_state.hr_series = None
    
    if stop_scan:
        st.session_state.scanning = False
    
    if st.session_state.scanning:
        progress_bar = st.progress(0)
        timer_text = st.empty()
    
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            frame_count = 0
            start_time = time.time()
            
            while st.session_state.scanning and (time.time() - start_time) < 30:
                ret, frame = cap.read()
                if not ret:
                    break
                
                rgb_values, face_bbox = extract_rgb_signal(frame)
                
                if rgb_values is not None:
                    st.session_state.rgb_history.append(rgb_values)
                    st.session_state.timestamp_history.append(time.time())
                    
                    if face_bbox is not None:
                        x, y, w, h = face_bbox
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (102, 126, 234), 3)
                        cv2.putText(frame, "Face Detected", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (102, 126, 234), 2)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                elapsed = time.time() - start_time
                progress = min(elapsed / 30, 1.0)
                progress_bar.progress(progress)
                timer_text.markdown(f'<div class="timer">{int(30 - elapsed)} seconds remaining</div>', unsafe_allow_html=True)
                
                frame_count += 1
                time.sleep(0.033)
            
            if time.time() - start_time >= 30:
                st.session_state.scanning = False
                st.session_state.scan_complete = True
                
                rgb_array = np.array(st.session_state.rgb_history)
                timestamps = np.array(st.session_state.timestamp_history)
                
                if len(rgb_array) > 100:
                    fps = len(rgb_array) / 30
                    heart_rate, hr_series = calculate_heart_rate(rgb_array, timestamps, fps)
                    st.session_state.hr_series = hr_series
                    st.session_state.scan_results['heart_rate'] = heart_rate
                    
                    hrv = calculate_hrv(hr_series) if hr_series is not None else None
                    if hrv:
                        st.session_state.scan_results['hrv'] = hrv
                    
                    breathing_rate = calculate_breathing_rate(rgb_array, fps)
                    st.session_state.scan_results['breathing_rate'] = breathing_rate
                    
                    stress_label, stress_score, stress_color = calculate_stress_level(
                        heart_rate, hrv, breathing_rate
                    )
                    st.session_state.scan_results['stress'] = {
                        'label': stress_label,
                        'score': stress_score,
                        'color': stress_color
                    }
                
                st.success("✅ Scan complete! View your results below.")
                st.rerun()
    except Exception as e:
        st.error(f"Camera error: {str(e)}")
    finally:
        if cap is not None:
            cap.release()

with col2:
    st.markdown("###  Your Health Metrics")
    
    if st.session_state.scan_complete and st.session_state.scan_results:
        results = st.session_state.scan_results
        
        if results.get('heart_rate'):
            hr = results['heart_rate']
            hr_color = "#ef4444" if hr > 100 else "#10b981" if hr >= 60 else "#f59e0b"
            hr_status = "Normal" if 60 <= hr <= 100 else "High" if hr > 100 else "Low"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {hr_color};">{hr:.0f}</div>
                <div class="metric-label">Heart Rate (BPM)</div>
                <div class="metric-trend">{hr_status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if results.get('hrv'):
            hrv_data = results['hrv']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{hrv_data.get('rmssd', 0):.0f}</div>
                <div class="metric-label">Heart Rate Variability (ms)</div>
                <div class="metric-trend">{hrv_data.get('status', 'Normal')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if results.get('breathing_rate'):
            br = results['breathing_rate']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{br:.1f}</div>
                <div class="metric-label">Breathing Rate (breaths/min)</div>
                <div class="metric-trend">{'Normal' if 12 <= br <= 20 else 'Elevated' if br > 20 else 'Low'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if results.get('stress'):
            stress = results['stress']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {stress['color']};">{stress['label']}</div>
                <div class="metric-label">Stress Level</div>
                <div class="metric-trend">Score: {stress['score']:.0f}/100</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("###  Health Summary")
        
        summary = []
        if results.get('heart_rate'):
            hr = results['heart_rate']
            if 60 <= hr <= 100:
                summary.append(" Heart rate is within normal range")
            elif hr > 100:
                summary.append(" Heart rate is elevated - consider relaxation techniques")
            else:
                summary.append(" Heart rate is lower than average - common in athletes")
        
        if results.get('stress'):
            stress = results['stress']
            if stress['label'] == "Low Stress":
                summary.append(" Low stress level detected - good mental state")
            elif stress['label'] == "Moderate Stress":
                summary.append(" Moderate stress detected - consider stress reduction activities")
            else:
                summary.append(" High stress detected - deep breathing recommended")
        
        if results.get('breathing_rate'):
            br = results['breathing_rate']
            if 12 <= br <= 20:
                summary.append(" Breathing rate is normal")
            else:
                summary.append(" Breathing rate is elevated - try slow, deep breaths")
        
        for item in summary:
            st.markdown(f"<div class='status-card'>{item}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("###  Recommendations")
        
        if results.get('stress') and results['stress']['label'] != "Low Stress":
            st.markdown("""
            **Stress Reduction Tips:**
            - Deep breathing exercises (4-7-8 technique)
            - 5-minute mindfulness meditation
            - Take a short walk
            - Stay hydrated
            """)
        
        if results.get('heart_rate') and results['heart_rate'] > 100:
            st.markdown("""
            **Heart Rate Management:**
            - Reduce caffeine intake
            - Practice slow, deep breathing
            - Ensure adequate sleep
            - Consider consulting a healthcare provider
            """)
        
        if st.button(" Download Health Report", use_container_width=True):
            report = f"""
            VitalLens Health Report
            ======================
            Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Metrics:
            - Heart Rate: {results.get('heart_rate', 0):.0f} BPM
            - HRV (RMSSD): {results.get('hrv', {}).get('rmssd', 0):.0f} ms
            - Breathing Rate: {results.get('breathing_rate', 0):.1f} breaths/min
            - Stress Level: {results.get('stress', {}).get('label', 'N/A')}
            
            Recommendations:
            {chr(10).join(summary)}
            
            Disclaimer: This is for wellness purposes only. Not a medical diagnostic tool.
            """
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"vitalens_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key="download_report"
            )
    else:
        st.info(" Click 'Start 30-Second Scan' to begin your health assessment")

# Visualization section
if st.session_state.scan_complete and len(st.session_state.rgb_history) > 0:
    st.markdown("---")
    st.markdown("###  Signal Analysis")
    
    rgb_array = np.array(st.session_state.rgb_history)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("RGB Signal (Raw)", "Heart Rate Signal (Filtered)", 
                        "FFT Spectrum", "Heart Rate Trend")
    )
    
    time_axis = np.linspace(0, 30, len(rgb_array))
    fig.add_trace(go.Scatter(x=time_axis, y=rgb_array[:, 1], 
                             name="Green Channel", line=dict(color='#10b981')),
                  row=1, col=1)
    
    green_signal = rgb_array[:, 1]
    green_signal = signal.detrend(green_signal)
    green_signal = (green_signal - np.mean(green_signal)) / np.std(green_signal)
    
    fps = len(rgb_array) / 30
    nyquist = fps / 2
    low = 0.8 / nyquist
    high = 3.0 / nyquist
    
    try:
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_signal = signal.filtfilt(b, a, green_signal)
        fig.add_trace(go.Scatter(x=time_axis, y=filtered_signal, 
                                 name="Filtered Signal", line=dict(color='#667eea')),
                      row=1, col=2)
    except:
        pass
    
    fft_vals = np.fft.fft(green_signal)
    fft_freq = np.fft.fftfreq(len(green_signal), 1/fps)
    magnitude = np.abs(fft_vals)
    
    positive_freq = fft_freq[:len(fft_freq)//2]
    positive_mag = magnitude[:len(magnitude)//2]
    
    fig.add_trace(go.Scatter(x=positive_freq, y=positive_mag, 
                             name="FFT Magnitude", line=dict(color='#f59e0b')),
                  row=2, col=1)
    
    if st.session_state.hr_series is not None and len(st.session_state.hr_series) > 0:
        fig.add_trace(go.Scatter(x=list(range(len(st.session_state.hr_series))), 
                                 y=st.session_state.hr_series,
                                 name="Heart Rate", line=dict(color='#ef4444')),
                      row=2, col=2)
    
    fig.update_layout(height=600, showlegend=True, 
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="Measurement", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>✨ VitalLens | AI-Powered Health Monitoring | No Wearable Required</p>
    <p style="font-size: 0.75rem;">Powered by Computer Vision & Signal Processing | For wellness purposes only - not a medical device</p>
</div>
""", unsafe_allow_html=True)
