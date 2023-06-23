import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the data
df = pd.read_csv('extracted_features.csv', sep='\t', on_bad_lines='skip')

# Convert classification labels to binary
df['classification'] = df['classification'].apply(lambda x: 1 if x.strip() == 'dga' else 0)

# Split the dataset into features and labels
X = df.drop(['domain', 'classification', 'length', 'english_word_count'], axis=1)
y = df['classification']

# Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
print("Accuracy: ", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=5)

# Print cross-validation scores
print('Cross-validation scores: ', scores)

# Print the mean of the cross-validation scores
print('Mean cross-validation score: ', scores.mean())

# Generate confusion matrix
cm = confusion_matrix(y_test, predictions)

# Calculate True Positives, False Positives, False Negatives, and True Negatives
TP = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]
TN = cm[0, 0]

# Print the results
print(f"True Positives: {TP}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"True Negatives: {TN}")

# Get feature importances
importances = model.feature_importances_

# Print feature importances
for feature, importance in zip(X.columns, importances):
    print(f"Feature: {feature}, Importance: {importance}")

# Create a dictionary with the evaluation results
results = {
    'Accuracy': accuracy_score(y_test, predictions),
    'Precision': precision_score(y_test, predictions),
    'Recall': recall_score(y_test, predictions),
    'F1-score': f1_score(y_test, predictions),
    'Mean cross-validation score': scores.mean(),
    'True Positives': TP,
    'False Positives': FP,
    'False Negatives': FN,
    'True Negatives': TN,
    **dict(zip(X.columns, importances))  # Add feature importances
}

# Convert the dictionary to a pandas DataFrame
results_df = pd.DataFrame([results])

# Save the DataFrame to a CSV file
results_df.to_csv('evaluation_results_Setting2_gradientboostingclassifier.csv', index=False)
