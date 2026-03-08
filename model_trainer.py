import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("Loading dataset...")
df = pd.read_csv('data/accidents_cleaned.csv')

le_weather = LabelEncoder()
le_road    = LabelEncoder()
le_day     = LabelEncoder()
le_state   = LabelEncoder()

df['Weather_Encoded'] = le_weather.fit_transform(df['Weather_Condition'])
df['Road_Encoded']    = le_road.fit_transform(df['Road_Type'])
df['Day_Encoded']     = le_day.fit_transform(df['Day_of_Week'])
df['State_Encoded']   = le_state.fit_transform(df['State'])

df['Is_Rush_Hour'] = df['Hour'].apply(lambda h: 1 if (7 <= h <= 9) or (16 <= h <= 18) else 0)
df['Is_Night']     = df['Hour'].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
df['Is_Weekend']   = df['Day_of_Week'].isin(['Saturday','Sunday']).astype(int)
df['Bad_Weather']  = df['Weather_Condition'].isin(['Heavy Rain','Snow','Fog','Thunderstorm','Hail']).astype(int)

FEATURES = [
    'Temperature_F','Wind_Speed_mph','Visibility_mi','Precipitation_in',
    'Humidity_pct','Pressure_in','Speed_Limit','Weather_Encoded','Road_Encoded',
    'Hour','Day_Encoded','Month','State_Encoded',
    'Junction','Traffic_Signal','Crossing','Stop','Amenity',
    'Is_Rush_Hour','Is_Night','Is_Weekend','Bad_Weather'
]

X = df[FEATURES]
y = df['Severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

print("Training Random Forest (200 trees)...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=15, min_samples_split=5,
    min_samples_leaf=2, max_features='sqrt',
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred   = rf.predict(X_test)
acc      = accuracy_score(y_test, y_pred)
cv       = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
feat_imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)

print(f"\nAccuracy:  {acc*100:.2f}%")
print(f"CV Mean:   {cv.mean()*100:.2f}% +/- {cv.std()*100:.2f}%")
print(f"\n{classification_report(y_test, y_pred)}")

os.makedirs('models', exist_ok=True)
joblib.dump(rf,         'models/rf_model.pkl')
joblib.dump(le_weather, 'models/le_weather.pkl')
joblib.dump(le_road,    'models/le_road.pkl')
joblib.dump(le_day,     'models/le_day.pkl')
joblib.dump(le_state,   'models/le_state.pkl')

meta = {
    'accuracy':  round(float(acc), 4),
    'cv_mean':   round(float(cv.mean()), 4),
    'cv_std':    round(float(cv.std()), 4),
    'features':  FEATURES,
    'weather_classes': list(le_weather.classes_),
    'road_classes':    list(le_road.classes_),
    'day_classes':     list(le_day.classes_),
    'state_classes':   list(le_state.classes_),
    'feature_importances': {k: round(float(v), 5) for k, v in feat_imp.items()},
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    'train_size': int(len(X_train)),
    'test_size':  int(len(X_test)),
}
with open('models/model_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("Model saved  ->  models/rf_model.pkl")
print("Meta saved   ->  models/model_meta.json")
print("\nDone! Now run:  streamlit run app.py")