# 🛡️ GAN-Powered Intrusion Detection System
```
AI-894: Design and Implementation of AI Systems
Created by: Abebual Zerihun Demilew
```
***
This advanced intrusion detection system leverages Conditional Generative Adversarial Networks (CGAN) to generate various synthetic IoT attack dataset and implements a XGBoost multi-class classification model to detect the type of intrusion.

## Demo App

![GAN-Powered IDS App](https://legendary-memory-v6gj9p4p65rhw66q-8501.app.github.dev/)

## Application Use Case

This app employs a sophisticate end-to-end workflow designed to enhance cybersecurity in the Internet of Things (IoT) domain. The application allow users:

**1. Generate synthetic IoT intrusion data:** At the heart of the process lies the utilization of advanced Conditional Generative Adversarial Networks (cGANs). These networks are adept at creating synthetic data that mimics real-world IoT intrusion scenarios, simulating realistic cybersecurity threats.

**2. Data Pre-processing:** Once the synthetic data is generated, the data is then undergoes pre-processing to align its structure with the format used during the training of the XGBoost multi-class classification model, ensuring consistency and accuracy.

**3. Machine Learning Model Inference:** The pre-processed data is then fed into a XGBoost multi-class classification machine learning model. This model is trained to recognize and categorize various types of IoT intrusion attacks, making it a powerful tool for cybersecurity.

**4. Post-Model Insights:** After the model makes its predictions, the application provides insights to understand the types of attacks and their characteristics. Its a critical step for cybersecurity professionals to take informed actions and strengthen IoT security measures.

