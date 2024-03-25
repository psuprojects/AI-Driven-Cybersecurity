import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time

from sklearn.preprocessing import StandardScaler
import joblib


# Page title
st.set_page_config(page_title='GAN-Powered Intrusion Detection System', page_icon='🛡️')
st.title('🛡️ GAN-Powered Intrusion Detection System')
st.header("AI-894: Design and Implementation of AI Systems")
st.markdown("Created by: Abebual Zerihun Demilew")

st.markdown("---")
st.markdown("""
      
            
    This advanced intrusion detection system leverages Conditional Generative Adversarial Networks (CGAN) to generate various synthetic IoT attack dataset and implements a XGBoost multi-class classification model to detect the type of intrusion. The CGAN model was trained on RT-IOT2022 data that can be found here: 

            
        
    https://archive.ics.uci.edu/dataset/942/rt-iot2022         
"""
)
st.markdown("---")

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app employs a sophisticate end-to-end workflow designed to enhance cybersecurity in the Internet of Things (IoT) domain. The application allow users:')
  st.info('**Generate synthetic IoT intrusion data:** At the heart of the process lies the utilization of advanced Conditional Generative Adversarial Networks (cGANs). These networks are adept at creating synthetic data that mimics real-world IoT intrusion scenarios, simulating realistic cybersecurity threats.')
  st.info('**Data Pre-processing:** Once the synthetic data is generated, the data is then undergoes pre-processing to align its structure with the format used during the training of the XGBoost multi-class classification model, ensuring consistency and accuracy.')
  st.info('**Machine Learning Model Inference:** The pre-processed data is then fed into a XGBoost multi-class classification machine learning model. This model is trained to recognize and categorize various types of IoT intrusion attacks, making it a powerful tool for cybersecurity.')
  st.info('**Post-Model Insights:** After the model makes its predictions, the application provides insights to understand the types of attacks and their characteristics. Its a critical step for cybersecurity professionals to take informed actions and strengthen IoT security measures.')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and 1. Simulate realistic Iot intrusion data 2. Upload generated data for pre-processing and Inference. As a result, display the attack type prediction results as well as insight on characterstics of the attack.')

  st.markdown('**Under the hood**')
  st.markdown('Data sets:')
  st.code('''- RT-IOT2022 data set
  ''', language='markdown')
  
  st.markdown('Libraries used:')
  st.code('''- mlflow==2.9.2
- category-encoders==2.6.3
- cloudpickle==2.0.0
- configparser==5.2.0
- holidays==0.38
- psutil==5.9.0
- scikit-learn==1.1.1
- typing-extensions==4.4.0
- xgboost==1.7.6
- tensorflow==2.15.0
- pandas==1.5.3
- Streamlit==1.32.2
  ''', language='markdown')


# Sidebar for accepting input parameters
# Sidebar: Data simulation
with st.sidebar:
    st.header('1. Simulate IoT Intrusion Data')
    n_samples = st.slider('Number of synthetic samples', 1, 1000, 100)
    if st.button('Generate Synthetic Data'):
        with st.spinner('Generating...'):
            cGAN_RT_IDS_synthesizer = CTGANSynthesizer.load("cGAN_RT_IDS_synthesizer.pkl")
            synthetic_data = cGAN_RT_IDS_synthesizer.sample(n_samples)
            synthetic_data.to_csv('synthetic_data.csv', index=False)
            st.success('Generated!')

            # Download button for synthetic data
            with open('synthetic_data.csv', 'rb') as f:
                st.download_button('Download Simulated Data CSV', f, file_name='synthetic_data.csv')

    # Sidebar: Data preprocessing
    st.header('2. Data Pre-processing')
    uploaded_file = st.file_uploader('Upload your CSV file', type=['csv'], key='file_uploader')
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write('Raw Data Preview:', data.head())

        # Data preprocessing
        scaler = StandardScaler()
        supported_cols = ["active_avg", "service_radius", "id_resp_p", "fwd_pkts_per_sec", "fwd_pkts_payload_avg", "id_orig_p", "flow_pkts_payload_tot", "service_dns", "fwd_header_size_tot", "service_dhcp", "flow_pkts_payload_min", "bwd_pkts_per_sec", "flow_pkts_per_sec", "active_min", "service_http", "fwd_header_size_max", "flow_pkts_payload_max", "bwd_pkts_payload_avg", "proto_tcp", "flow_SYN_flag_count", "service_mqtt", "fwd_header_size_min", "bwd_header_size_max", "fwd_pkts_payload_min", "fwd_iat_min", "service_unspec", "proto_udp", "flow_pkts_payload_avg", "fwd_URG_flag_count", "proto_icmp", "service_ssl", "flow_iat_min", "fwd_pkts_payload_tot", "fwd_last_window_size", "service_irc", "flow_FIN_flag_count", "payload_bytes_per_second", "fwd_init_window_size", "fwd_pkts_payload_max", "service_ntp"]
        data[supported_cols] = scaler.fit_transform(data[supported_cols])

        # Display preprocessed data
        st.write('Preprocessed Data Preview:', data[supported_cols].head())

    # Sidebar: Model Inference
    st.header('3. Model Inference')
    if st.button('Run Model Inference') and uploaded_file is not None:
        with st.spinner('Loading model and predicting...'):
            model = joblib.load('cGAN_RT_IDS_classifier.pkl')
            predictions = model.predict(data[supported_cols])
            st.write('Prediction Results:', predictions)

    # Sidebar: Post-Model Insights
    st.header('4. Post-Model Insights')
    if st.button('Generate Insights'):
        st.write('Insight generation is not implemented yet.')

st.warning('👈 Simulate IoT intrusion data to get started!')
