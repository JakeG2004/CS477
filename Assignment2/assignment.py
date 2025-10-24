import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Read in the CSVs using pandas
df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')

def make_cleaned_data(df):
    # Create the departure year, month, day
    dates = df['FL_DATE'].astype(str).str.split('-')
    df['DEPARTURE_YEAR'] = dates.str[0]
    df['DEPARTURE_MONTH'] = dates.str[1]
    df['DEPARTURE_DAY'] = dates.str[2]
    df.drop(columns=['FL_DATE', 'MONTH', 'DAY_OF_MONTH'], axis=1, inplace=True)

    # Create the scheduled departure time columns
    df['Scheduled_DEP_EST'] = pd.to_datetime(df['Scheduled_DEP_EST'], errors='coerce')
    df['SCHEDULED_DEPARTURE_HOUR'] = df['Scheduled_DEP_EST'].dt.hour
    df['SCHEDULED_DEPARTURE_MINUTE'] = df['Scheduled_DEP_EST'].dt.minute

    # Create the scheduled arrival time columns
    df['Scheduled_ARR_EST'] = pd.to_datetime(df['Scheduled_ARR_EST'], errors='coerce')
    df['SCHEDULED_ARRIVAL_HOUR'] = df['Scheduled_ARR_EST'].dt.hour
    df['SCHEDULED_ARRIVAL_MINUTE'] = df['Scheduled_ARR_EST'].dt.minute

    # Isolate the time that the airplane is scheduled to arrived at origin airport
    df['Scheduled_ARR_Ori'] = pd.to_datetime(df['Scheduled_ARR_Ori'], errors='coerce')
    df['SCHEDULED_ORIGIN_ARRIVAL_HOUR'] = df['Scheduled_ARR_Ori'].dt.hour
    df['SCHEDULED_ORIGIN_ARRIVAL_MINUTE'] = df['Scheduled_ARR_Ori'].dt.minute

    # Isolate the time that the airplane actually arrived at origin airport
    df['Actual_ARR_dt_Ori'] = pd.to_datetime(df['Actual_ARR_dt_Ori'], errors='coerce')
    df['ACTUAL_ORIGIN_ARRIVAL_HOUR'] = df['Actual_ARR_dt_Ori'].dt.hour
    df['ACTUAL_ORIGIN_ARRIVAL_MINUTE'] = df['Actual_ARR_dt_Ori'].dt.minute

    # Use one-hot encoding on MKT-CARRIER, OP-CARRIER, ORIGIN, DEST, FAA_CLASS, day_of_week
    df = pd.get_dummies(df, columns=['OP_CARRIER', 'ORIGIN', 'DEST', 'FAA_class', 'day_of_week'])

    # Use on hot encoding with drop first
    df = pd.get_dummies(df, columns=['late_airjet_when_turnaround_within_180'], drop_first=True)

    # Remove unnecessary columns
    df.drop(columns=['Scheduled_DEP', 'Scheduled_ARR_Local', 'MKT_CARRIER', 'CRS_DEP_1hrpre', 'CRS_DEP_1hrpost', 'Scheduled_ARR_Ori', 'Actual_ARR_dt_Ori', 'Scheduled_DEP_EST', 'Scheduled_ARR_EST', 'Scheduled_ARR_Local'], axis=1, inplace=True)

    return df

# Clean the data
df_train_cleaned = make_cleaned_data(df_train)
df_test_cleaned = make_cleaned_data(df_test)

# Create the datasets
X_train = df_train_cleaned.drop('DEP_DEL15', axis=1)
X_test = df_test_cleaned.drop('ID', axis=1)
y_train = df_train_cleaned['DEP_DEL15']

# Align columns between training and test sets
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Create the predictions
y_pred = model.predict(X_test)

# Apply threshold
y_pred_binary = (y_pred >= 0.5).astype(int)

submission = pd.DataFrame({'ID': pd.Series(df_test_cleaned['ID']), 'DEP_DEL15':pd.Series(y_pred_binary)})
submission.to_csv('submission.csv', index=False)
print("Submission file successfully created!")