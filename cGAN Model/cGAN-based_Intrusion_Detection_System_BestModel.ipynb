# Databricks notebook source
# MAGIC %md
# MAGIC # Developing and Evaluating Intrusion Detection Systems (IDS)

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.models import Model
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials, SparkTrials, space_eval
from hyperopt.early_stop import no_progress_loss
import mlflow
from mlflow.models.signature import infer_signature
import sdv
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import torch
# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Print versions
print(mlflow.__version__)
print(tf.__version__)
print(sdv.version.public)

# MLflow configuration
mlflow.tensorflow.autolog()


# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data Preparation 
# MAGIC

# COMMAND ----------

# Load dataset 
processed_file_path = '/dbfs/FileStore/m332479/GANs_forCyberSecurity/processed_RT_IOT2022.csv'
df = pd.read_csv(processed_file_path , index_col=0).reset_index()
df = df.dropna()

# Single Table Metadata API


metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

print(metadata)
df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialize MLFlow Experments

# COMMAND ----------

# COMMAND ----------
mlflow.end_run()
# Initializes an MLflow experiment for tracking machine learning tasks. 
# It checks if the specified experiment already exists and creates it if not, storing the experiment ID.
dbutils.widgets.text("mlflow_exp_root_path","/Users/m332479@azg.pwus.us/ml_experments")

from mlflow.tracking import MlflowClient
mlflow_exp_root_path = dbutils.widgets.get("mlflow_exp_root_path")
client = MlflowClient()

## Test if experiment already exists
exp_name = f"{mlflow_exp_root_path}/cGAN_bestmodel"
if exp_name in [x.name for x in client.search_experiments()]:
    exp = mlflow.set_experiment(exp_name)
    experiment_id = exp.experiment_id
else:
    ## Create an experiment for runs started from a repo notebook
    experiment_id = client.create_experiment(f"{mlflow_exp_root_path}/cGAN_bestmodel")
experiment_id


# COMMAND ----------

study = 'march22'

run_name = f'cGAN_bestmodel{study}'
exp_id = experiment_id

# COMMAND ----------

CGAN_RT_IOT2022 = CTGANSynthesizer(
    metadata, # required
    enforce_rounding=False,
    epochs=1000,
    batch_size = 500,
    discriminator_dim = (512, 512),
    discriminator_decay = 2e-4,
    discriminator_steps = 1,
    embedding_dim = 128,
    generator_decay = 1e-4,
    generator_dim = (512, 512),
    generator_lr = 2e-4,
    log_frequency = True,
    pac = 10,
    verbose=True
)
mlflow.log_param("epochs", 1000)
mlflow.log_param("batch_size", 500)
mlflow.log_param("discriminator_dim", (512, 512))
mlflow.log_param("generator_dim", (512, 512))
mlflow.log_param("discriminator_decay", 2e-4)
mlflow.log_param("generator_decay", 2e-4)
mlflow.log_param("generator_lr", 1e-4)

print("Best Parameters: ", CGAN_RT_IOT2022.get_parameters())
print("---------------------------------------------------")
print("Start training...")
print("---------------------------------------------------")
CGAN_RT_IOT2022.fit(df)
CGAN_RT_IOT2022.save(
    filepath='/dbfs/FileStore/m332479/GANs_forCyberSecurity/models/cGAN_RT_IDS_synthesizer.pkl'
)

loss_df = CGAN_RT_IOT2022.get_loss_values()
# Convert all the tensor values to floats
loss_df['Generator Loss'] = loss_df['Generator Loss'].apply(lambda x: x.item())
loss_df['Discriminator Loss'] = loss_df['Discriminator Loss'].apply(lambda x: x.item())


# Set 'Epochs' as the index
loss_df.set_index('Epoch', inplace=True)
  
# Save the DataFrame as a CSV or log it as an artifact in MLflow
loss_df.to_csv("/dbfs/FileStore/m332479/GANs_forCyberSecurity/loss_df_cgan.csv")
mlflow.log_artifact("/dbfs/FileStore/m332479/GANs_forCyberSecurity/loss_df_cgan.csv")
mlflow.log_metric("Discriminator Loss", loss_df['Generator Loss'].mean())
mlflow.log_metric("Generator Loss", loss_df['Discriminator Loss'].mean())


# Plot the loss.
fig, ax = plt.subplots(figsize=(10, 4))
loss_df.plot(ax=ax)
plt.title('cGAN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['G Loss', 'D Loss'])
plt.grid(True)

# Make sure to use a correct path to save the plot.
plot_path =  "/dbfs/FileStore/m332479/GANs_forCyberSecurity/plots/loss_plot_sdv.png"
plt.savefig(plot_path)

# COMMAND ----------


# Make sure to use a correct path to save the plot.
plot_path =  "/dbfs/FileStore/m332479/GANs_forCyberSecurity/plots/loss_plot_cGAN.png"
plt.savefig(plot_path)
mlflow.log_artifact("/dbfs/FileStore/m332479/GANs_forCyberSecurity/plots/loss_plot_cGAN.png")
mlflow.log_artifact("/dbfs/FileStore/m332479/GANs_forCyberSecurity/models/cGAN_RT_IDS_synthesizer.pkl")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC In the context of a Conditional Generative Adversarial Network (CGAN), both the generator and the discriminator have specific roles:
# MAGIC
# MAGIC `Generator`: It tries to generate data that is indistinguishable from real data, given some condition or class label.
# MAGIC
# MAGIC `Discriminator`: It tries to distinguish between real data and the fake data generated by the generator.
# MAGIC During training, these two networks are in a sort of competition: the generator tries to improve its data generation to fool the discriminator, while the discriminator tries to get better at distinguishing real from fake.
# MAGIC
# MAGIC `Loss Functions`: Typically, the loss functions for both networks are designed to measure how well each is performing its task.
# MAGIC
# MAGIC `Generator Loss`: This often reflects how poorly the generator is at fooling the discriminator. A high loss means the discriminator can easily tell its data is fake, while a lower loss means the generator is producing more believable data.
# MAGIC
# MAGIC `Discriminator Loss`: This reflects how well the discriminator is at identifying real and fake data. A high loss would mean the discriminator is often fooled by the generator, while a lower loss would mean it's proficient at telling real from fake.
# MAGIC
# MAGIC `Negative Loss Values`: In the case of CGANs, or GANs in general, negative loss values can occur depending on the design of the loss functions. Some loss functions (like Wasserstein Loss) can naturally lead to negative values because they measure a kind of "distance" between distributions. Other reasons for negative losses can include:
# MAGIC Logarithms in Loss Functions: Traditional GAN loss functions involve logarithms, which can lead to negative values since the logarithm of a number between 0 and 1 is negative. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sampling Realistic Synthetic Data

# COMMAND ----------


from sdv.single_table import CTGANSynthesizer

cGAN_RT_IDS_synthesizer = CTGANSynthesizer.load(
    filepath="/dbfs/FileStore/m332479/GANs_forCyberSecurity/models/cGAN_RT_IDS_synthesizer.pkl",
)


# COMMAND ----------

cGAN_RT_IDS_synthetic_data= cGAN_RT_IDS_synthesizer.sample(num_rows=500_000)
cGAN_RT_IDS_synthetic_data

# COMMAND ----------

cGAN_RT_IDS_synthetic_data.Attack_type.value_counts()

# COMMAND ----------

df.Attack_type.value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation

# COMMAND ----------

from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
from sdv.evaluation.single_table import get_column_plot

# 1. perform basic validity checks
diagnostic = run_diagnostic(df, cGAN_RT_IDS_synthetic_data, metadata)
diagnostic

# COMMAND ----------

# MAGIC %md
# MAGIC Interpreting the Score:
# MAGIC The score should be 100%. The diagnostic report checks for basic data validity and data structure issues. You should expect the score to be perfect for any of the default SDV synthesizers.
# MAGIC
# MAGIC What's Included?
# MAGIC
# MAGIC **Data Validity:** Basic validity checks for each of the columns:
# MAGIC 1. Primary keys must always be unique and non-null
# MAGIC 2. Continuous values in the synthetic data must adhere to the min/max range in the real data
# MAGIC 3. Discrete values in the synthetic data must adhere to the same categories as the real data.
# MAGIC
# MAGIC **Structure:** Checks to ensure the real and synthetic data have the same column names
# MAGIC

# COMMAND ----------

# 2. measure the statistical similarity
quality_report = evaluate_quality(df, cGAN_RT_IDS_synthetic_data, metadata)
quality_report

# COMMAND ----------

# MAGIC %md
# MAGIC **Data Quality**
# MAGIC
# MAGIC The Quality Report checks for statistical similarity between the real and the synthetic data. Use this to discover which patterns the synthetic data has captured from the real data.
# MAGIC
# MAGIC Interpreting the Score: Your score will vary from 0% to 100%. This value tells you how similar the synthetic data is to the real data.
# MAGIC A 100% score means that the patterns are exactly the same. For example, if you compared the real data with itself (identity), the score would be 100%.
# MAGIC A 0% score means the patterns are as different as can be. This would entail that the synthetic data purposefully contains anti-patterns that are opposite from the real data.
# MAGIC Any score in the middle can be interpreted along this scale. For example, a score of 80% means that the synthetic data is about 80% similar to the real data — about 80% of the trends are similar. The quality score is expected to vary, and you may never achieve exactly 100% quality. That's ok! The SDV synthesizers are designed to estimate patterns, meaning that they may smoothen, extrapolate, or noise certain parts of the data. 
# MAGIC
# MAGIC **Column Shapes:** The statistical similarity between the real and synthetic data for single columns of data. This is often called the marginal distribution of each column.
# MAGIC
# MAGIC **Column Pair Trends** The statistical similarity between the real and synthetic data for pairs of columns. This is often called the correlation or bivariate distributions of the columns.
# MAGIC

# COMMAND ----------

# Loop through each column in the DataFrame


# Retrieve the column names from the real_data DataFrame
column_names = df.columns[:20]  

for column_name in column_names:
    # Generate the column plot for each column
    fig = get_column_plot(
        real_data=df,
        synthetic_data=cGAN_RT_IDS_synthetic_data,
        metadata=metadata,
        column_name=column_name
    )
    
    # Display the figure
    fig.show()    
    # To prevent open figures from accumulating, close the figure
    plt.close('all')



# COMMAND ----------

# Loop through each column in the DataFrame


# Retrieve the column names from the real_data DataFrame
column_names = df.columns[20:30]  

for column_name in column_names:
    # Generate the column plot for each column
    fig = get_column_plot(
        real_data=df,
        synthetic_data=cGAN_RT_IDS_synthetic_data,
        metadata=metadata,
        column_name=column_name
    )
    
    # Display the figure
    fig.show()    
    # To prevent open figures from accumulating, close the figure
    plt.close('all')



# COMMAND ----------

 # save the data as a CSV
cGAN_RT_IDS_synthetic_data.to_csv("/dbfs/FileStore/m332479/GANs_forCyberSecurity/synthetic_data.csv", index=False)

# COMMAND ----------

# Find the number of rows for the smallest class
smallest_class_size = cGAN_RT_IDS_synthetic_data['Attack_type'].value_counts().min()

# Sample from each class
balanced_synthetic_data = cGAN_RT_IDS_synthetic_data.groupby('Attack_type').apply(lambda x: x.sample(smallest_class_size)).reset_index(drop=True)
balanced_synthetic_data['Attack_type'].value_counts()

# COMMAND ----------

balanced_synthetic_data.to_csv("/dbfs/FileStore/m332479/GANs_forCyberSecurity/balanced_synthetic_data.csv", index=False)

# COMMAND ----------


