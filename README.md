# Predictive Cyber Defence World Model

Flow + packet telemetry -> 10-second states -> LSTM world model -> K-step forecast -> infiltration probability -> ATT&CK-oriented stage -> explainability.

Run:
`streamlit run app/app.py`

Configuration:
- Window: 10s
- History: 20 states
- Forecast horizon: 5
- Seed: 42

The system is an offline dataset-based prototype. ATT&CK stage labels are an explicit mapping because CIC-IDS2018 does not provide a complete campaign timeline.
