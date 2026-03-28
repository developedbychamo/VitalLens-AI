# ✨ VitalLens AI: Zero-Hardware Biometric Intelligence

**VitalLens AI** transforms standard RGB webcams into medical-grade health sensors. Built during **LovHack Season 2**, this project uses computer vision and signal processing to democratize health monitoring.

##  The Mission
Clinical-grade monitoring shouldn't require expensive wearables. VitalLens AI leverages **Remote Photoplethysmography (rPPG)** to detect micro-vascular changes in the skin, providing real-time health insights from a simple 30-second video feed.

##  Key Features
- **Heart Rate Monitoring:** Real-time BPM detection.
- **Stress Level Analysis:** Intelligence derived from HRV and heart rate trends.
- **Heart Rate Variability (HRV):** Tracking autonomic nervous system health.
- **Breathing Rate:** Monitoring respiratory patterns through subtle color shifts.

##  Technical Stack (M2-Optimized)
- **Core Logic:** Python + OpenCV (Face tracking & ROI extraction)
- **Signal Processing:** SciPy (Butterworth Bandpass Filtering & Fast Fourier Transform)
- **Visualization:** Plotly + Streamlit (Interactive health dashboards)
- **Architecture:** JSON-based data-interchange for modular analytics.

##  How it Works
1. **Face Detection:** OpenCV isolates the forehead and cheek regions.
2. **Signal Extraction:** Raw RGB pixel values are gathered at 30 FPS.
3. **Filtering:** A 4th-order Butterworth filter isolates the pulse frequency (0.8Hz - 3.0Hz).
4. **Quantification:** FFT converts time-domain signals into heart rate peaks.

##  Hardware Compatibility
Optimized specifically for **Apple Silicon (M2)** to ensure low-latency, real-time processing while keeping all biometric data local and private.

---
*Developed as a solo project by a Data Science Undergraduate at General Sir John Kotelawala Defence University (KDU).*
