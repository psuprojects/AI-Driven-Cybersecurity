# 🛡️ GAN-Powered Intrusion Detection System
```
AI-894: Design and Implementation of AI Systems
Created by: Abebual Zerihun Demilew
```
***
## Introduction 

In the realm of cybersecurity, the proliferation of Internet of Things (IoT) devices coupled with sophisticated network attack methodologies presents an ongoing challenge for defenders. To address this challenge, this project proposes the development and implementation of a Generative Adversarial Network (GAN)-based Intrusion Detection System (IDS) tailored for multi-class classification. Leveraging the RT-IoT2022 dataset, which encapsulates diverse IoT devices and attack scenarios, the proposed IDS aims to achieve a micro-averaged F1 score of at least 95% in discerning between nine types of attack patterns and three types of normal patterns. The project objectives encompass the development of a robust GAN architecture, mitigation of GAN-specific challenges, and seamless integration with existing security infrastructures. The methodology entails a comprehensive data analytics workflow, encompassing data collection, preprocessing, feature selection, GAN model design, training, evaluation, and refinement, culminating in integration and testing in simulated environments. The significance of this research lies in its potential to address the evolving cyber threats efficiently, thereby enhancing cybersecurity measures, protecting personal data, and mitigating financial losses for businesses. Moreover, it contributes to the advancement of AI in practical applications, particularly in the cybersecurity domain, paving the way for future developments in defensive mechanisms against cyberattacks.


## Demo App
This advanced intrusion detection system leverages Conditional Generative Adversarial Networks (CGAN) to generate various synthetic IoT attack dataset and implements a XGBoost multi-class classification model to detect the type of intrusion.

![GAN-Powered IDS App](https://legendary-memory-v6gj9p4p65rhw66q-8501.app.github.dev/)

## Application Use Case

This app employs a sophisticate end-to-end workflow designed to enhance cybersecurity in the Internet of Things (IoT) domain. The application allow users:

**1. Generate synthetic IoT intrusion data:** At the heart of the process lies the utilization of advanced Conditional Generative Adversarial Networks (cGANs). These networks are adept at creating synthetic data that mimics real-world IoT intrusion scenarios, simulating realistic cybersecurity threats.

**2. Data Pre-processing:** Once the synthetic data is generated, the data is then undergoes pre-processing to align its structure with the format used during the training of the XGBoost multi-class classification model, ensuring consistency and accuracy.

**3. Machine Learning Model Inference:** The pre-processed data is then fed into a XGBoost multi-class classification machine learning model. This model is trained to recognize and categorize various types of IoT intrusion attacks, making it a powerful tool for cybersecurity.

**4. Post-Model Insights:** After the model makes its predictions, the application provides insights to understand the types of attacks and their characteristics. Its a critical step for cybersecurity professionals to take informed actions and strengthen IoT security measures.


