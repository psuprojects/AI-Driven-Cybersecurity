import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.evaluation.single_table import get_column_plot
import joblib
import torch
import databricks.automl_runtime
import mlflow
import lightgbm
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder as SklearnOneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
import shap

# Page title
st.set_page_config(page_title='GAN-Powered Intrusion Detection System', page_icon='🛡️')
st.title('🛡️ GAN-Powered Intrusion Detection System')
st.header("Design and Implementation of AI Systems")
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
  st.code('''- RT-IOT2022 data set:
        https://archive.ics.uci.edu/dataset/942/rt-iot2022
  ''', language='markdown')
  
  st.markdown('Libraries used:')
  st.code(''' 
        - pandas==1.5.3
        - numpy==1.26.4
        - matplotlib==3.7.0
        - seaborn==0.13.2
        - Streamlit==1.32.2
        - joblib==1.3.2
        - scikit-learn==1.4.1.post1
        - mlflow==2.9.2
        - lightgbm==4.1.0
        - shap==0.45.0
        - category-encoders==2.6.3
        - configparser==5.2.0
        - holidays==0.38
        - psutil==5.9.0
        - typing-extensions==4.4.0
        - tensorflow==2.15.0
        - cffi==1.15.1
        - cloudpickle==2.2.1
        - configparser==5.2.0
        - defusedxml==0.7.1
        - graphviz==0.20.3
  ''', language='markdown')

# Sidebar for accepting input parameters
# Sidebar: Data simulation
with st.sidebar:
    st.header('1. Simulate IoT Intrusion Data')
    n_samples = st.slider('Number of synthetic samples', 1, 10000, 1000)
    if st.button('Generate Synthetic Data'):
        with st.spinner('Generating...'):
            cGAN_RT_IDS_synthesizer = CTGANSynthesizer.load("cGAN_IDS_synthesizer.pkl")
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

        if st.button('Run Diagnostic'):
            orginal_df = pd.read_csv('original_df.csv', index_col=0)
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(orginal_df)
            # perform basic validity checks
            diagnostic = run_diagnostic(orginal_df, data , metadata, verbose=False)
            # Display the preprocessed data
            fig = diagnostic.get_visualization(property_name='Data Validity')
            st.write('Diagnostic Fig:', fig.show())
            st.write('Diagnostic Report:', diagnostic.get_details(property_name='Data Validity'))
            from sdv.evaluation.single_table import get_column_plot

            fig = get_column_plot(
                real_data=orginal_df,
                synthetic_data=data,
                metadata=metadata,
                column_name='payload_bytes_per_second'
            )
    
            st.write('Column Plot:', fig.show())
        attacks = ['DOS_SYN_Hping', 'Thing_Speak', 'ARP_poisioning', 'MQTT_Publish', 'NMAP_UDP_SCAN', 'NMAP_XMAS_TREE_SCAN', 'Wipro_bulb']
        data = data[data['Attack_type'].isin(attacks)]
        st.write('Raw Data Preview:', data.head())

        # Data preprocessing
        target_col = "Attack_type"
        
        from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
        supported_cols = ["active_avg", "service_radius", "id_resp_p", "fwd_pkts_per_sec", "fwd_pkts_payload_avg", "id_orig_p", "flow_pkts_payload_tot", "service_dns", "fwd_header_size_tot", "service_dhcp", "flow_pkts_payload_min", "bwd_pkts_per_sec", "flow_pkts_per_sec", "active_min", "service_http", "fwd_header_size_max", "flow_pkts_payload_max", "bwd_pkts_payload_avg", "proto_tcp", "flow_SYN_flag_count", "service_mqtt", "fwd_header_size_min", "bwd_header_size_max", "fwd_pkts_payload_min", "fwd_iat_min", "service_unspec", "proto_udp", "flow_pkts_payload_avg", "fwd_URG_flag_count", "proto_icmp", "service_ssl", "flow_iat_min", "fwd_pkts_payload_tot", "fwd_last_window_size", "service_irc", "flow_FIN_flag_count", "payload_bytes_per_second", "fwd_init_window_size", "fwd_pkts_payload_max", "service_ntp"]
        col_selector = ColumnSelector(supported_cols)

        # Boolean columns
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer
        from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


        bool_imputers = []

        bool_pipeline = Pipeline(steps=[
            ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
            ("imputers", ColumnTransformer(bool_imputers, remainder="passthrough")),
            ("onehot", SklearnOneHotEncoder(handle_unknown="ignore", drop="first")),
        ])

        bool_transformers = [("boolean", bool_pipeline, ["proto_udp", "service_http", "service_irc", "fwd_URG_flag_count", "service_radius", "proto_icmp", "service_ssl", "service_dns", "proto_tcp", "service_mqtt", "service_dhcp", "service_ntp", "service_unspec"])]
        
        # Numerical columns
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler

        num_imputers = []
        num_imputers.append(("impute_mean", SimpleImputer(), ["active_avg", "active_min", "bwd_header_size_max", "bwd_pkts_payload_avg", "bwd_pkts_per_sec", "flow_FIN_flag_count", "flow_SYN_flag_count", "flow_iat_min", "flow_pkts_payload_avg", "flow_pkts_payload_max", "flow_pkts_payload_min", "flow_pkts_payload_tot", "flow_pkts_per_sec", "fwd_URG_flag_count", "fwd_header_size_max", "fwd_header_size_min", "fwd_header_size_tot", "fwd_iat_min", "fwd_init_window_size", "fwd_last_window_size", "fwd_pkts_payload_avg", "fwd_pkts_payload_max", "fwd_pkts_payload_min", "fwd_pkts_payload_tot", "fwd_pkts_per_sec", "id_orig_p", "id_resp_p", "payload_bytes_per_second", "proto_icmp", "proto_tcp", "proto_udp", "service_dhcp", "service_dns", "service_http", "service_irc", "service_mqtt", "service_ntp", "service_radius", "service_ssl", "service_unspec"]))

        numerical_pipeline = Pipeline(steps=[
            ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
            ("imputers", ColumnTransformer(num_imputers)),
            ("standardizer", StandardScaler()),
        ])

        numerical_transformers = [("numerical", numerical_pipeline, ["active_avg", "id_resp_p", "fwd_pkts_payload_avg", "fwd_pkts_per_sec", "id_orig_p", "service_radius", "flow_pkts_payload_tot", "service_dns", "fwd_header_size_tot", "service_dhcp", "flow_pkts_payload_min", "bwd_pkts_per_sec", "flow_pkts_per_sec", "active_min", "service_http", "fwd_header_size_max", "flow_pkts_payload_max", "bwd_pkts_payload_avg", "flow_SYN_flag_count", "proto_tcp", "fwd_header_size_min", "service_mqtt", "bwd_header_size_max", "fwd_pkts_payload_min", "fwd_iat_min", "service_unspec", "flow_pkts_payload_avg", "proto_udp", "fwd_URG_flag_count", "proto_icmp", "service_ssl", "flow_iat_min", "fwd_pkts_payload_tot", "fwd_last_window_size", "service_irc", "flow_FIN_flag_count", "payload_bytes_per_second", "fwd_init_window_size", "fwd_pkts_payload_max", "service_ntp"])]
        
        from sklearn.compose import ColumnTransformer

        transformers = bool_transformers + numerical_transformers

        preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0) 
        
        # Separate target column from features
        y = data[target_col]

        data = data.drop([target_col], axis=1)
        
        
        pipeline_val = Pipeline([
            ("column_selector", col_selector),
            ("preprocessor", preprocessor),
        ])
        pipeline_val.fit(data, y)
        

        # Manually set the feature names for the 'cast_type' step, assuming it just converts types without changing the number or order of features
        cast_type_feature_names = ["proto_udp", "service_http", "service_irc", "fwd_URG_flag_count", "service_radius", "proto_icmp", "service_ssl", "service_dns", "proto_tcp", "service_mqtt", "service_dhcp", "service_ntp", "service_unspec"]

        # Get the feature names for the one-hot encoded features
        onehot_feature_names = pipeline_val.named_steps['preprocessor'].transformers_[0][1].named_steps['onehot'].get_feature_names_out(cast_type_feature_names)

        # Combine the one-hot encoded feature names with the numerical feature names
        # Assuming numerical features are passed through as-is (since they are just scaled), append their original names
        numerical_feature_names = ["active_avg", "id_resp_p", "fwd_pkts_payload_avg", "fwd_pkts_per_sec", "id_orig_p", "service_radius", "flow_pkts_payload_tot", "service_dns", "fwd_header_size_tot", "service_dhcp", "flow_pkts_payload_min", "bwd_pkts_per_sec", "flow_pkts_per_sec", "active_min", "service_http", "fwd_header_size_max", "flow_pkts_payload_max", "bwd_pkts_payload_avg", "flow_SYN_flag_count", "proto_tcp", "fwd_header_size_min", "service_mqtt", "bwd_header_size_max", "fwd_pkts_payload_min", "fwd_iat_min", "service_unspec", "flow_pkts_payload_avg", "proto_udp", "fwd_URG_flag_count", "proto_icmp", "service_ssl", "flow_iat_min", "fwd_pkts_payload_tot", "fwd_last_window_size", "service_irc", "flow_FIN_flag_count", "payload_bytes_per_second", "fwd_init_window_size", "fwd_pkts_payload_max", "service_ntp"]

        # Combine all the feature names
        all_feature_names = list(onehot_feature_names) + numerical_feature_names

        data_processed = pipeline_val.transform(data)
        data_processed = pd.DataFrame(data_processed, columns=all_feature_names)
        
        # Display the preprocessed data
        st.write('Preprocessed Data Preview:', data_processed.head())


    # Sidebar: Model Inference
    st.header('3. Model Inference')
    if st.button('Run Model Inference') and uploaded_file is not None:
        with st.spinner('Loading model and predicting...'):
            model = joblib.load('lightGBM_multiclass_classifier.joblib')
            predictions = model.predict(data_processed)
            #label_encoder = LabelEncoder()
            #y_encoded = label_encoder.fit_transform(y)
            sample_prediction = pd.DataFrame({
                'Actual Labels': y,
                'Predictions': predictions
            })
            # Display the preprocessed data
            st.write('Sample Predictions:', sample_prediction.head())
            cm = confusion_matrix(y, predictions)

            # Normalize the confusion matrix
            cm_normalized = normalize(cm, axis=1, norm='l1')
            
            # Dictionary mapping from encoded label integers to string representations
            label_dict = {
                1: "DOS_SYN_Hping",
                5: "Thing_Speak",
                0: "ARP_poisioning",
                2: "MQTT_Publish",
                4: "NMAP_XMAS_TREE_SCAN",
                3: "NMAP_UDP_SCAN",
                6: "Wipro_bulb"
            }

           # Creating a list of labels in the correct order
            labels = [label_dict[i] for i in sorted(label_dict.keys())]

            # Plotting the confusion matrix
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f',
                             xticklabels=labels, yticklabels=labels)

            # Labels, title, and ticks
            label_font = {'size': '12'}
            ax.set_xlabel('Predicted labels', fontdict=label_font)
            ax.set_ylabel('True labels', fontdict=label_font)
            ax.set_title('Normalized Confusion Matrix', fontdict={'size': '15'})
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels, rotation=0)

            # Use Streamlit's pyplot to render the matplotlib figure
            st.pyplot(plt)

    # Display the button and the plot in the Streamlit interface
    st.header('4. Post-Model Insights')
    if st.button('Generate Insights'):
        with st.spinner('Generating insights...'):
            
            # *Load your model outside of your function to avoid reloading it on every interaction
            #model = joblib.load('lightGBM_multiclass_classifier.joblib')
            #explainer = shap.TreeExplainer(model)
            

            #shap_values = explainer.shap_values(data_processed)
         
            # *Creating a DataFrame for each class and concatenating them
            #df_list = []
            #for class_index, class_shap_values in enumerate(shap_values):
                #if len(all_feature_names) != len(class_shap_values[0]):
                    #raise ValueError(f"Feature names count does not match SHAP values count: {len(all_feature_names)} vs {len(class_shap_values[0])}")
                #class_df = pd.DataFrame(class_shap_values, columns=all_feature_names)
                # *Use the label_dict to get the corresponding label name for the class index
                #class_label = label_dict[class_index]
                #class_df['class'] = class_label  # Assign the label name instead of the class index
                #df_list.append(class_df)

            # *Concatenate all DataFrames
            #shap_df = pd.concat(df_list, ignore_index=True)
            #shap_df.to_csv('shap_df.csv', index=False)
            #st.success('Shap Values Calculated!')

            # Download button for synthetic data
            #with open('shap_df.csv', 'rb') as f:
                #st.download_button('Download Calculated Shap Values', f, file_name='shap_df.csv')
         
            shap_df = pd.read_csv('shap_df.csv')
            # Group by 'class' and then compute the mean absolute SHAP value for each feature within each class
            grouped_shap = shap_df.groupby('class').mean().abs()

            # Loop through each class to create a plot
            for class_label, shap_values in grouped_shap.iterrows():
                # Sort the features by importance
                shap_values_sorted = shap_values.sort_values(ascending=False)
                plt.figure(figsize=(10, 6))
                shap_values_sorted.plot(kind='bar', title=f"Feature Importance for {class_label}")
                plt.ylabel('Mean Absolute SHAP Value')
                #st.write('Attack Type Level Insight:', plt.show())
                st.pyplot(plt)
          

st.warning('👈 Simulate IoT intrusion data to get started!')
