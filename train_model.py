import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load CSV
data = pd.read_csv('data/gesture_data.csv')

print("Data shape:", data.shape)
print("Columns and types:\n", data.dtypes)
print(data.head())

# The **first column** is the label column
y = data.iloc[:, 0]        # first column (labels)
X = data.iloc[:, 1:]       # rest 63 columns (features)

# Convert features to float
X = X.astype(float)

# Train KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)

# Save model
with open('gesture_knn_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Training complete and model saved as 'gesture_knn_model.pkl'.")

