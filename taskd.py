import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

df = pd.read_csv('train.csv')

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for column in df.columns:
    df[column] = label_encoder.fit_transform(df[column])

model = RandomForestClassifier()
model.fit(df.drop('class', axis=1), df['class'])

importance = pd.Series(model.feature_importances_, index=df.columns[:-1]).sort_values(ascending=False)
print(f"Importance:{importance}")

X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

with open('mushroom_model_system.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved as mushroom_model_system.pkl")
