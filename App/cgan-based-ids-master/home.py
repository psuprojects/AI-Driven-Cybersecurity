import streamlit as st

st.set_page_config(page_title="GAN-Powered Intrusion Detection System", page_icon="📖", layout="wide")

st.header("AI-894: Design and Implementation of AI Systems")


st.markdown("---")
st.markdown("""
    Created by: Abebual Zerihun Demilew
    This advanced intrusion detection system leverages Conditional Generative Adversarial Networks (CGAN) to generate various synthetic IoT attack dataset and implements a lightGBM multi-class classification model to detect the type of intrusion. The CGAN model was trained RT-IOT2022 data that can be found here: 

            
        
    https://archive.ics.uci.edu/dataset/942/rt-iot2022         
"""
)
st.markdown("---")