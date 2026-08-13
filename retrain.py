import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

df = pd.read_csv('datasets/cleaned_yield.csv')
print("Columns:", df.columns.tolist())

# Convert yield units
if 'yield_tonnes_per_ha' not in df.columns:
    df['yield_tonnes_per_ha'] = df['yield_hg_per_ha'] / 10000

# Encode categorical columns
le_crop = LabelEncoder()
le_country = LabelEncoder()
df['crop_encoded'] = le_crop.fit_transform(df['crop'])
df['country_encoded'] = le_country.fit_transform(df['country'])

# Use only 10% of data
df = df.sample(frac=0.1, random_state=42)
print("Sample size:", df.shape)

X = df[['crop_encoded','country_encoded','year',
        'rainfall_mm','pesticides_tonnes','avg_temp']]
y = df['yield_tonnes_per_ha']

X_train,X_test,y_train,y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=10, max_depth=5,
    random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

pickle.dump(model, open('models/yield_model.pkl','wb'))
print('Done! New model saved.')
size = os.path.getsize('models/yield_model.pkl')
print(f'New size: {size/1024/1024:.1f} MB')