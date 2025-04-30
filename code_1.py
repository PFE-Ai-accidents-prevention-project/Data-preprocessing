
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Load the preprocessed data
df = pd.read_csv('modified_merged_dataset_to_predict.csv')

# If neither 'Accident_type' nor 'Accident_type ' exists, there is a data issue 
# Check your earlier pre-processing to verify.

# Select relevant columns (Date_fix and Accident_type )
df = df[['Date_fix', 'Title_Incident_Report']]  # Adjusted to 'Accident_type'

# Convert 'Date_fix' to datetime objects
df['Date_fix'] = pd.to_datetime(df['Date_fix'], format='%d/%m/%Y', errors='coerce')
df.dropna(subset=['Date_fix'], inplace=True)

# Encode incident types into numerical labels
le = LabelEncoder()
df['Title_Incident_Report'] = le.fit_transform(df['Title_Incident_Report'])  # Adjusted to 'Accident_type'

# Set 'Date_fix' as index and sort the DataFrame
df = df.set_index('Date_fix').sort_index()

# Create the ARIMA model
#p, d, and q are order of model, you may need to adjust this
model = ARIMA(df['Title_Incident_Report'], order=(5, 1, 0))  # Adjusted to 'Accident_type'
model_fit = model.fit()

# Split data into training and testing sets (crucial for evaluation)
train_size = int(len(df) * 0.8)  # 80% for training, adjust as needed
train_data, test_data = df[0:train_size], df[train_size:len(df)]

# Make predictions
#Adjust the steps for how far you want to predict in the future
predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1]) 

# Convert predictions back to original labels
predicted_labels = le.inverse_transform(predictions.astype(int))

# Print predicted incident types
print(f"predicted_labels :{predicted_labels}") 

# Calculate evaluation metrics
accuracy = accuracy_score(test_data['Title_Incident_Report'], predictions.astype(int))
precision = precision_score(test_data['Title_Incident_Report'], predictions.astype(int), average='weighted')
recall = recall_score(test_data['Title_Incident_Report'], predictions.astype(int), average='weighted')

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
