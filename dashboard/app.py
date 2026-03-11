import streamlit as st
import pandas as pd
import glob

files = glob.glob("../data/experiments/*.csv")

st.title("WiFi Human Detection")

if files:
    latest = sorted(files)[-1]
    df = pd.read_csv(latest)

    st.line_chart(df)
else:
    st.write("No CSI data yet")
