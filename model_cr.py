import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import re
# Step 1: Prepare the Dataset

# Define the dataset
crop_data = pd.read_csv('F:/Shihab MS/Thesis Wrok/Code/Dataset/Crop_and_fertilizer_dataset.csv')

# Assuming 'df' is your DataFrame
crop_data = crop_data.drop(columns=['District_Name', 'Link'])
def rename_columns(data):
    new_columns = []
    for column in data.columns:
        new_column_name = re.sub(r'(?<!^)(?=[A-Z][a-z])', '_', column).lower()
        new_columns.append(new_column_name)
    data.columns = new_columns
rename_columns(crop_data)
print(crop_data.columns)



# Convert categorical data to numerical using the mappings provided
soil_color_map = {
    "Black": 1, "Red": 2, "Dark Brown": 3, "Reddish Brown": 4,
    "Light Brown": 5, "Medium Brown": 6
}

fertilizer_map = {
    "Urea": 1, "DAP": 2, "MOP": 3, "19:19:19 NPK": 4, "SSP": 5,
    "Magnesium Sulphate": 6, "10:26:26 NPK": 7, "50:26:26 NPK": 8,
    "Chelated Micronutrient": 9, "12:32:16 NPK": 10, "Ferrous Sulphate": 11,
    "13:32:26 NPK": 12, "Ammonium Sulphate": 13, "10:10:10 NPK": 14,
    "Hydrated Lime": 15, "White Potash": 16, "20:20:20 NPK": 17,
    "18:46:00 NPK": 18, "Sulphur": 19
}

crop_map = {
    "Sugarcane": 1, "Wheat": 2, "Cotton": 3, "Jowar": 4, "Maize": 5,
    "Rice": 6, "Groundnut": 7, "Tur": 8, "Grapes": 9, "Ginger": 10,
    "Urad": 11, "Moong": 12, "Gram": 13, "Turmeric": 14, "Soybean": 15,
    "Masoor": 16, "Banana": 17, "Sunflower": 18, "Pigeon Pea": 19, "Cabbage": 20
}

crop_data['soil_color'] = crop_data['soil_color'].map(soil_color_map)
crop_data['fertilizer'] = crop_data['fertilizer'].map(fertilizer_map)
crop_data['crop'] = crop_data['crop'].map(crop_map)


###
# Select numerical columns
numerical_columns = crop_data.select_dtypes(include=[np.number]).columns

# Function to identify outliers using IQR method
def find_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return outliers

# Dictionary to store outliers for each column
outliers_dict = {}

for col in numerical_columns:
    outliers = find_outliers(crop_data, col)
    outliers_dict[col] = outliers
    print(f'Number of outliers in {col}: {len(outliers)}')

# Display outliers for each column (if needed)
for col, outliers in outliers_dict.items():
    if not outliers.empty:
        print(f'\nOutliers in {col}:\n{outliers}')

# Optional: Displaying first few rows of outliers for each column
for col, outliers in outliers_dict.items():
    if not outliers.empty:
        print(f'\nOutliers in {col} (first 5 rows):')
        print(outliers.head())

# Remove outliers from the dataset
for col in numerical_columns:
    Q1 = crop_data[col].quantile(0.25)
    Q3 = crop_data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    crop_data = crop_data[(crop_data[col] >= lower_bound) & (crop_data[col] <= upper_bound)]

print(f'Dataset shape after removing outliers: {crop_data.shape}')



# Step 2: Split the Dataset
X = crop_data.drop('crop', axis=1)  # Features (excluding the target 'crop')
y = crop_data['crop']  # Target variable (crop)

# Step 3: Apply Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Handle Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Step 5: Split the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 6: Train the Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)

# Step 8: Save the Model and Scaler
with open('crop_model.pkl', 'wb') as model_file:
    pickle.dump(rf_model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model saved as 'crop_model.pkl'")
print("Scaler saved as 'scaler.pkl'")
