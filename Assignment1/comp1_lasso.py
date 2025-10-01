import numpy as np
import csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso

def read_csv(filename):
    return_list = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        
        # Skip the header
        next(reader)

        for row in reader:
            return_list.append(row)

    return return_list

# age,sex,bmi,children,smoker,region
def clean_data(data, testing):
    offset = int(testing)

    return_data = []
    for row in data:

        age = float(row[0 + offset])
        sex = (row[1 + offset] == 'male')
        bmi = float(row[2 + offset])
        children = float(row[3 + offset])
        smoker = (row[4 + offset] == 'yes')
        northwest = (row[5 + offset] == 'northwest')
        southeast = (row[5 + offset] == 'southeast')
        northeast = (row[5 + offset] == 'northeast')
        southwest = (row[5 + offset] == 'southwest')

        return_data.append([
            age,
            sex, 
            bmi, 
            children,
            smoker, 
            northwest,
            southeast,
            northeast,
            southwest,
            bmi * age,
            (not sex and children > 0),
            ])

    return return_data

# Import the data from the csv
raw_training_data = read_csv('train.csv')
raw_testing_data = read_csv('test.csv')

# Pull our ML targets from the raw training data
training_targets = [[float(row[6])] for row in raw_training_data]

# Pull our IDs from the raw testing data
testing_ids = [[int(row[0])] for row in raw_testing_data]

# Clean the data
training_data = clean_data(raw_training_data, testing=False)
testing_data =  clean_data(raw_testing_data, testing=True)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 10))#StandardScaler()
scaled_training_data = scaler.fit_transform(np.array(training_data, dtype=float))
scaled_testing_data = scaler.transform(np.array(testing_data, dtype=float))

# Train a linear model
model = LinearRegression()

# Set up k-fold cross validation
from sklearn.model_selection import KFold
Kf = KFold(n_splits=5, shuffle=True)

# Create a full cross-validation loop
from sklearn.metrics import mean_absolute_error

mae_score = []
fold_number = 1

for train_idx, test_idx in Kf.split(scaled_training_data):
    # Generate training and testing datasets
    X_train, y_train = scaled_training_data[train_idx], np.array(training_targets)[train_idx]
    X_test, y_test = scaled_training_data[test_idx], np.array(training_targets)[test_idx]

    # Fit the model to the new data
    model.fit(X_train, y_train)

    # Make predictions
    preds = model.predict(X_test)

    # Calculate our MAE value
    mae = mean_absolute_error(y_test, preds)
    mae_score.append(mae)

    print(f"MAE at iteration {fold_number} is {mae}")

    fold_number += 1

"""
model.fit(scaled_training_data, training_targets)

# Generate our predictions
target_predictions = model.predict(scaled_testing_data)

# Write the predictions back to a CSV file
with open('submission_lasso.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['ID', 'charges'])
    for idx, row in enumerate(target_predictions, start=0):
        writer.writerow([testing_ids[idx][0], row])"""