from statsmodels.tsa.arima.model import ARIMA
import itertools
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
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

# Split data into training and testing sets (crucial for evaluation)
train_size = int(len(df) * 0.8)  # 80% for training, adjust as needed
train_data, test_data = df[0:train_size], df[train_size:len(df)]

p = d = q = range(0, 3)  # Adjust range as needed
pdq = list(itertools.product(p, d, q))

best_aic = float("inf")
best_order = None

for order in pdq:
    try:
        model = ARIMA(train_data['Title_Incident_Report'], order=order)
        model_fit = model.fit()
        if model_fit.aic < best_aic:
            best_aic = model_fit.aic
            best_order = order
    except:
        continue

print(f"Best ARIMA order: {best_order}")

result = adfuller(df['Title_Incident_Report'])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")


model = SARIMAX(df['Title_Incident_Report'], order=(p, d, q))
model_fit = model.fit()


# Make predictions
#Adjust the steps for how far you want to predict in the future
predictions = model_fit.predict(start=test_data.index[0], end=test_data.index[-1]) 

# Convert predictions back to original labels
predicted_labels = le.inverse_transform(predictions.astype(int))

accuracy = accuracy_score(test_data['Title_Incident_Report'], predictions.astype(int))
precision = precision_score(test_data['Title_Incident_Report'], predictions.astype(int), average='weighted')
recall = recall_score(test_data['Title_Incident_Report'], predictions.astype(int), average='weighted')