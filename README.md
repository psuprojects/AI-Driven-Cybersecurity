# 🛡️cGAN-Powered Intrusion Detection System
```
AI-894: Design and Implementation of AI Systems
Created by: Abebual Zerihun Demilew
Instructor: Dr. Youakim Badr

```
***
## Introduction 

In the realm of cybersecurity, the proliferation of Internet of Things (IoT) devices coupled with sophisticated network attack methodologies presents an ongoing challenge for defenders. To address this challenge, this project proposes the development and implementation of a Generative Adversarial Network (GAN)-based Intrusion Detection System (IDS) tailored for multi-class classification. Leveraging the RT-IoT2022 dataset, which encapsulates diverse IoT devices and attack scenarios, the proposed IDS aims to achieve a micro-averaged F1 score of at least 95% in discerning between nine types of attack patterns and three types of normal patterns. The project objectives encompass the development of a robust GAN architecture, mitigation of GAN-specific challenges, and seamless integration with existing security infrastructures. The methodology entails a comprehensive data analytics workflow, encompassing data collection, preprocessing, feature selection, GAN model design, training, evaluation, and refinement, culminating in integration and testing in simulated environments. The significance of this research lies in its potential to address the evolving cyber threats efficiently, thereby enhancing cybersecurity measures, protecting personal data, and mitigating financial losses for businesses. Moreover, it contributes to the advancement of AI in practical applications, particularly in the cybersecurity domain, paving the way for future developments in defensive mechanisms against cyberattacks.  <br>

This app employs a sophisticate end-to-end workflow designed to enhance cybersecurity in the Internet of Things (IoT) domain. The application allow users:

**1. Generate synthetic IoT intrusion data:** At the heart of the process lies the utilization of advanced Conditional Generative Adversarial Networks (cGANs). These networks are adept at creating synthetic data that mimics real-world IoT intrusion scenarios, simulating realistic cybersecurity threats.

**2. Data Pre-processing:** Once the synthetic data is generated, the data is then undergoes pre-processing to align its structure with the format used during the training of the XGBoost multi-class classification model, ensuring consistency and accuracy.

**3. Machine Learning Model Inference:** The pre-processed data is then fed into a XGBoost multi-class classification machine learning model. This model is trained to recognize and categorize various types of IoT intrusion attacks, making it a powerful tool for cybersecurity.

**4. Post-Model Insights:** After the model makes its predictions, the application provides insights to understand the types of attacks and their characteristics. Its a critical step for cybersecurity professionals to take informed actions and strengthen IoT security measures.


## References
***
This advanced intrusion detection system leverages Conditional Generative Adversarial Networks (CGAN) to generate various synthetic IoT attack dataset and implements a lightGBM multi-class classification model to detect the type of intrusion. The app demo is available 
![here](https://legendary-memory-v6gj9p4p65rhw66q-8501.app.github.dev/). <br>
A presentation of the work is avaialble [here on YouTube](https://www.youtube.com/watch?v=DOJrRgg0XOU).

## Installation
***
To ensure smooth operation and compatibility, it's crucial that you install the specific versions of the libraries and packages listed below. This guide will walk you through the steps to set up your environment to run the project's code. <br>
### Prerequisites
Before you begin, ensure that you have Python installed on your system. This project requires Python 3.8 or higher. You can download and install the latest Python version from python.org.

### Setting Up Your Environment
It's recommended to use a virtual environment for this project to avoid conflicts with other packages or projects. If you're not familiar with virtual environments in Python, you can learn more about them ![here](https://docs.python.org/3/tutorial/venv.html).

### Installing Required Packages
With your virtual environment activated, install the required packages using the provided requirements.txt file. This file lists all the necessary libraries and their specific versions to ensure compatibility. <br>
Execute the following command to install the required packages:

```
pip install -r requirements.txt
```





