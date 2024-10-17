print("Loading Modules")
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
cat_variables = ['PetType', 'Breed', 'Color']
x = pd.get_dummies(x, columns=cat_variables, prefix=cat_variables)
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

print(" Split data for training/test")
report['Data']['Train/test split'] = {}
x_train, x_test, y_train, y_test = train_test_split(x, target, test_size=0.2, random_state=42)
report['Data']['Train/test split']['Train individuals'] = len(x_train)
report['Data']['Train/test split']['Test individuals'] = len(x_test)


print("Models creation")
report['Models'] = {}

print(" RandomForest")
report['Models']['RandomForest'] = {}
print("  Find best estimators")
n_range = range(50, 70)
scores = []
for n in n_range:
    classifier = RandomForestClassifier(n_estimators=n, random_state=13)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    scores.append(accuracy_score(y_test, y_pred))
plt.plot(n_range, scores)
plt.xlabel('Value of n_estimators for RandomForestClassifier')
plt.ylabel('Testing Accuracy')
fig_path = 'plot_BestEstimators.png'
plt.savefig(f"docs/{fig_path}")
report['Models']['RandomForest']['Best estimators'] = fig_path
print("   Build model")
RFclassifier = RandomForestClassifier(n_estimators=140, random_state=42)
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