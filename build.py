print("Loading Modules")
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import json
import utils

report = {}

print("Importing data")
report['Data'] = {}
df = pd.read_csv("data/pet_adoption_data.csv")
filename = 'df_fulldata.png'
utils.save_dataframe_head_as_png(df, f'docs/{filename}')
report['Data']['Raw'] = filename

print("Start Pre-processing")
report['Data']['Pre-Processing'] = {}

print(" Get descriptors")
x = df.drop(["PetID", "AdoptionLikelihood"], axis=1)
filename = 'df_descriptors.png'
utils.save_dataframe_head_as_png(x, f'docs/{filename}')
report['Data']['Pre-Processing']['Descriptors'] = filename

print(" Get target")
target = df['AdoptionLikelihood']
filename = 'df_target.png'
utils.save_dataframe_head_as_png(target, f'docs/{filename}')
report['Data']['Pre-Processing']['Target'] = filename

print(" Scale data")
numerical_variables = ['AgeMonths','WeightKg','TimeInShelterDays','AdoptionFee']
scaler = StandardScaler()
x[numerical_variables] = scaler.fit_transform(x[numerical_variables])
filename = 'df_standard_scaler.png'
utils.save_dataframe_head_as_png(x, f'docs/{filename}')
report['Data']['Pre-Processing']['Scaling'] = filename

print(" Encode")
label_encoders = {}
cat_variables = ['PetType', 'Breed', 'Color']
#x = pd.get_dummies(x, columns=cat_variables, prefix=cat_variables)
for col in cat_variables:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])
    label_encoders[col] = le
    report['Data']['Pre-Processing']['Encoding'] = f'{col} : {list(le.classes_)}'
encoder = OrdinalEncoder(categories=[['Small', 'Medium', 'Large']])
x['Size'] = encoder.fit_transform(x[['Size']])
filename = 'df_encode.png'
utils.save_dataframe_head_as_png(x, f'docs/{filename}')
report['Data']['Pre-Processing']['Encoding'] = filename

print(" Set 0/1 to Bool")
bool_variables = ['Vaccinated', 'HealthCondition', 'PreviousOwner']
x[bool_variables] = x[bool_variables].astype(bool)
filename = 'df_bool.png'
utils.save_dataframe_head_as_png(x, f'docs/{filename}')
report['Data']['Pre-Processing']['Set 0/1 to Bool'] = filename

# Feature selection based on random forest importances
selected_features = ['Size', 'AgeMonths', 'Vaccinated', 'HealthCondition', 'AdoptionFee', 'WeightKg', 'TimeInShelterDays', 'Breed']
selected_x = x[selected_features]

print("Models creation")
report['Models'] = {}
report['Models']['Selected Features'] = list(selected_features)
print(" Split data for training/test")
x_train, x_test, y_train, y_test = train_test_split(selected_x, target, test_size=0.2, random_state=42)
report['Models']['Train individuals'] = len(x_train)
report['Models']['Test individuals'] = len(x_test)
print(" RandomForest")
report['Models']['RandomForest'] = {}
print("  Build model")
RFclassifier = RandomForestClassifier(random_state=42)
RFclassifier.fit(x_train, y_train)
y_pred = RFclassifier.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
fig_path = 'plot_rf_conf_matrix.png'
plt.savefig(f"docs/{fig_path}")
plt.clf()
report['Models']['RandomForest']['Confusion Matrix'] = fig_path
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
report['Models']['RandomForest']['Accuracy'] = '{:.2f} %'.format(accuracy)
report['Models']['RandomForest']['Precision'] = '{:.2f} %'.format(precision)
report['Models']['RandomForest']['Sensibilite'] = '{:.2f} %'.format(recall)
report['Models']['RandomForest']['F1-Score'] = f1

print(" Naive Bayes")
report['Models']['Naive Bayes'] = {}
print("  Build model")
nb = GaussianNB()
nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
fig_path = 'plot_nb_conf_matrix.png'
plt.savefig(f"docs/{fig_path}")
plt.clf()
report['Models']['Naive Bayes']['Confusion Matrix'] = fig_path
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
report['Models']['Naive Bayes']['Accuracy'] = '{:.2f} %'.format(accuracy)
report['Models']['Naive Bayes']['Precision'] = '{:.2f} %'.format(precision)
report['Models']['Naive Bayes']['Sensibilite'] = '{:.2f} %'.format(recall)
report['Models']['Naive Bayes']['F1-Score'] = f1

with open('docs/report.json', 'w') as file:
    file.write(json.dumps(report))