
import streamlit as st
import pandas as pd, numpy as np, joblib, tensorflow as tf
from pathlib import Path

MODEL_DIR=Path("models")
model=tf.keras.models.load_model(MODEL_DIR/"cyber_world_model.keras")
prep=joblib.load(MODEL_DIR/"world_model_preprocessor.pkl")
scaler=prep["scaler"]; features=prep["features"]; stages=prep["stage_classes"]
SEQ=prep["sequence_length"]; H=prep["forecast_horizon"]

st.set_page_config(page_title="Cyber World Model",layout="wide")
st.title("Predictive Cyber Defence — Network World Model")
st.caption("Offline temporal network-state forecasting")

file=st.file_uploader("Upload timestamped network-state CSV",type=["csv"])
if file:
    df=pd.read_csv(file)
    missing=[c for c in features if c not in df.columns]
    if missing: st.error("Missing required features: "+", ".join(missing[:10])); st.stop()
    X=scaler.transform(df[features].replace([np.inf,-np.inf],np.nan).fillna(0))
    if len(X)<SEQ: st.error(f"Need at least {SEQ} states."); st.stop()

    seq=X[-SEQ:].astype("float32"); rows=[]
    for k in range(H):
        ps,pa,pt=model.predict(seq[None],verbose=0)
        ns=ps[0,0]; prob=float(pa[0,0,0]); sid=int(np.argmax(pt[0,0]))
        rows.append({"Future Window":k+1,"Infiltration Probability":prob,"Predicted Stage":stages[sid]})
        seq=np.concatenate([seq[1:],ns[None]],axis=0)

    out=pd.DataFrame(rows)
    st.subheader("K-step forecast"); st.dataframe(out,use_container_width=True)
    st.subheader("Infiltration probability timeline")
    st.line_chart(out.set_index("Future Window")["Infiltration Probability"])
    m=out.iloc[out["Infiltration Probability"].argmax()]
    st.metric("Maximum forecast probability",f"{m['Infiltration Probability']:.1%}")
    st.write("Highest-risk predicted stage:",m["Predicted Stage"])
    st.info("ATT&CK stage is an explicit dataset-label mapping, not a complete campaign timeline.")
