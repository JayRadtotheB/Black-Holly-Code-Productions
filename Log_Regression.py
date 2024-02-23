import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the CSV file into a DataFrame
file_path = "/Users/jbrow/Documents/Education/ECSU/CSC 460 +/WA_Fn-UseC_-HR-Employee-Attrition.csv"
data = pd.read_csv(file_path)

# Map categorical variables to numerical values
category_columns = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobSatisfaction',
                     'PerformanceRating', 'RelationshipSatisfaction', 'WorkLifeBalance']

label_encoder = LabelEncoder()
for col in category_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Assuming 'Attrition' is your target variable (1 if left, 0 if still with the company)
X = data.drop('Attrition', axis=1)
y = data['Attrition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_report_str = classification_report(y_test, predictions)

# Print the evaluation results
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report_str)
