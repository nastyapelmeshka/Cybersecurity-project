import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the data
df = pd.read_csv('extracted_features.csv', sep='\t', on_bad_lines='skip')

# Convert classification labels to binary
df['classification'] = df['classification'].apply(lambda x: 1 if x.strip() == 'dga' else 0)

# Split the dataset into features and labels
X = df.drop(['domain', 'classification'], axis=1)
y = df['classification']

# Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = KNeighborsClassifier(n_neighbors=5)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
print("Accuracy: ", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
