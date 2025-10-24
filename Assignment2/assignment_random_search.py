from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

# Read in the CSVs using pandas
df_train = pd.read_csv('./train.csv')
df_test = pd.read_csv('./test.csv')

def make_cleaned_data(df):
    # Create the departure year, month, day
    dates = df['FL_DATE'].astype(str).str.split('-')
    df['DEPARTURE_YEAR'] = dates.str[0].astype(int)
    df['DEPARTURE_MONTH'] = dates.str[1].astype(int)
    df['DEPARTURE_DAY'] = dates.str[2].astype(int)

    # Remove duplicate information
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

    # Use one-hot encoding on MKT_CARRIER, OP-CARRIER, ORIGIN, DEST, FAA_CLASS, day_of_week
    df = pd.get_dummies(df, columns=['OP_CARRIER', 'ORIGIN', 'DEST', 'FAA_class', 'day_of_week', 'MKT_CARRIER'])

    # Use on hot encoding with drop first
    df = pd.get_dummies(df, columns=['late_airjet_when_turnaround_within_180'], drop_first=True)

    # Delta Arrival time (interaction term)
    df['DT_ARRIVAL_HOUR'] = df['SCHEDULED_ORIGIN_ARRIVAL_HOUR'] - df['ACTUAL_ORIGIN_ARRIVAL_HOUR']
    df['DT_ARRIVAL_MINUTE'] = df['SCHEDULED_ORIGIN_ARRIVAL_MINUTE'] - df['ACTUAL_ORIGIN_ARRIVAL_MINUTE']

    # Time on tarmac is squared if the high risk flag is true
    df['scheduled_Turnarnd'] = np.where(
        df['late_airjet_when_turnaround_within_180_1'],  # condition column
        df['scheduled_Turnarnd'] ** 2,                 # value if True
        df['scheduled_Turnarnd']                       # value if False
    )

    # Remove unnecessary columns
    df.drop(columns=['Scheduled_DEP', 'Scheduled_ARR_Local', 'CRS_DEP_1hrpre', 'CRS_DEP_1hrpost', 'Scheduled_ARR_Ori', 'Actual_ARR_dt_Ori', 'Scheduled_DEP_EST', 'Scheduled_ARR_EST', 'Scheduled_ARR_Local'], axis=1, inplace=True)

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

# Base model
model = XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)


# Hyperparameter grid
param_dist = {
    'n_estimators': [100, 200, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.3, 0.5],
    'scale_pos_weight': [1, 2, 5, 10]
}

# F1 scorer
f1_scorer = make_scorer(f1_score)

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Randomized search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=40,             # number of random combinations to test
    scoring=f1_scorer,
    cv=skf,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# Run randomized search
random_search.fit(X_train, y_train)

print("Best parameters found:", random_search.best_params_)
print("Best F1 score:", random_search.best_score_)

# Retrieve the best model
best_model = random_search.best_estimator_

# Fit the best model on all training data
best_model.fit(X_train, y_train)

# Predict on the test data
y_pred = best_model.predict(X_test)

# Write submission
submission = pd.DataFrame({
    'ID': df_test_cleaned['ID'],
    'DEP_DEL15': y_pred
})
submission.to_csv('submission.csv', index=False)
print("Submission file successfully created with tuned model!")