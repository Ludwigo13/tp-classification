print("Loading Modules")
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

print("Importing data")
df = pd.read_csv("data/pet_adoption_data.csv")
print(df.head())


print("Start Pre-processing")

print(" Get descriptors")
x = df.drop(["PetID", "AdoptionLikelihood"], axis=1)
print(x.head())

print(" Get target")
target = df['AdoptionLikelihood']
print(target.head())

print(" Scale data")
numerical_variables = ['AgeMonths','WeightKg','TimeInShelterDays','AdoptionFee']
scaler = StandardScaler()
x[numerical_variables] = scaler.fit_transform(x[numerical_variables])
print(x.head())

print(" Encode")
cat_variables = ['PetType', 'Breed', 'Color']
x = pd.get_dummies(x, columns=cat_variables, prefix=cat_variables)
encoder = OrdinalEncoder(categories=[['Small', 'Medium', 'Large']])
x['Size'] = encoder.fit_transform(x[['Size']])
print(x.head())

print(" Set 0/1 to bool")
bool_variables = ['Vaccinated', 'HealthCondition', 'PreviousOwner']
x[bool_variables] = x[bool_variables].astype(bool)
print(x.head())

print(" Split data for training/test")
x_train, x_test, y_train, y_test = train_test_split(x, target, test_size=0.2, random_state=42)
print(f"  Train individuals: {len(x_train)}")
print(f"  Test individuals: {len(x_test)}")


print("Models creation")

print(" RandomForest")
RFclassifier = RandomForestClassifier(n_estimators=140, random_state=42)
RFclassifier.fit(x_train, y_train)
y_pred = RFclassifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"  Accuracy is {accuracy}")