import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("https://storage.googleapis.com/kagglesdsdata/datasets/6723424/10827884/Lung%20Cancer%20Dataset.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250412%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250412T194104Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=890b0accd389941d07372f8dd3d179bd4b962690ec7efa49e2f14d57e9286abd00efc1c6f1e9c8fec64b3b7ae77b8feb4fc8cf85747d2af73e45f64a6b312b42fbaf3ccd7fda9c25629fe3fbc9ea474cdbf4a1522ef5b4442d5359dad140876da620a0a67b7c0602a7312fc5b6453a3e1605454b2b662caa2ed221f9835f8ca47070bde1ca5cd1725a25df029d6390d1f49928a2b2a9340ebd99abd83ba6a41dfc88ccaaf9bfcb7819db280ef46e66a78cec0a32a66ede6b603525843ae0bdd9687692fdf94fb4a7dd9a487d53fdea322d579f2dc9179b391283fb2c0200617fa62f9dfbaf6f0b8515758929f1c5ca665cfa4550f26bcf3888118eb2a9d40027")

#use pearson correlation matrix to first figure out if there's any correlation between attributes and pulmonary disease
df['PULMONARY_DISEASE'] = df['PULMONARY_DISEASE'].str.strip().str.upper().map({"NO" : 0, "YES" : 1})
df1 = df[['AGE', 'GENDER', 'SMOKING', 'ALCOHOL_CONSUMPTION',
          'SMOKING_FAMILY_HISTORY', 'FAMILY_HISTORY','FINGER_DISCOLORATION', 'THROAT_DISCOMFORT', 'CHEST_TIGHTNESS', 'PULMONARY_DISEASE']]

df2 = df[['BREATHING_ISSUE', 'OXYGEN_SATURATION', 'MENTAL_STRESS', 'EXPOSURE_TO_POLLUTION', 'LONG_TERM_ILLNESS',
          'ENERGY_LEVEL', 'IMMUNE_WEAKNESS', 'STRESS_IMMUNE','PULMONARY_DISEASE']]

print(df.head())

plt.figure(figsize = (10,6))
sns.heatmap(df1.corr(),annot = True)
#sns.heatmap(df2.corr(),annot = True)

from sklearn.feature_selection import mutual_info_classif

X = df.drop('PULMONARY_DISEASE', axis = 1)
y = df[['PULMONARY_DISEASE']]

info = mutual_info_classif(X,y)
mi_df = pd.DataFrame({ "attributes" : X.columns, "mi" : info}).sort_values(by = "mi", ascending = False)

print(mi_df)

top_mi = mi_df.head(10)
sns.barplot(x="mi", y="attributes", data=top_mi)
plt.title("Top 10 Features by Mutual Information")
plt.show()

#red_df = df[['SMOKING','SMOKING_FAMILY_HISTORY','BREATHING_ISSUE','THROAT_DISCOMFORT','ENERGY_LEVEL','STRESS_IMMUNE','FAMILY_HISTORY','LONG_TERM_ILLNESS','PULMONARY_DISEASE']]
red_df = df
print(red_df)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X = df.drop(columns=['PULMONARY_DISEASE'])
y = df['PULMONARY_DISEASE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

jl_proj = GaussianRandomProjection(n_components=10, random_state=42)

X_train_jl = jl_proj.fit_transform(X_train_scaled)
X_test_jl = jl_proj.transform(X_test_scaled)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_jl, y_train)

y_pred = rf_clf.predict(X_test_jl)
accuracy = accuracy_score(y_test, y_pred)

print(f"Random Forest Accuracy with JL Projection: ", accuracy)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

rf_clf = RandomForestClassifier(n_estimators=200, random_state=45)

rf_clf.fit(X_train_scaled, y_train)

rf_train_preds = rf_clf.predict(X_train_scaled)
rf_test_preds = rf_clf.predict(X_test_scaled)

gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=45)
gb_clf.fit(rf_train_preds.reshape(-1, 1), y_train)

y_pred = gb_clf.predict(rf_test_preds.reshape(-1, 1))

accuracy = accuracy_score(y_test, y_pred)

print(f"Random Forest with Gradient Boosting: ", accuracy)

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

raw_df = red_df.copy()

X = raw_df.drop(columns=['PULMONARY_DISEASE'])
y = raw_df['PULMONARY_DISEASE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
nb_pred = nb_clf.predict(X_test)
nb_acc1 = nb_clf.score(X_test, y_test)

svm_clf = SVC(kernel='rbf', random_state=45)
svm_clf.fit(X_train, y_train)
svm_pred = svm_clf.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

rf = RandomForestClassifier(n_estimators=100, random_state=45)
rf.fit(X_train, y_train)
rf_acc = rf.score(X_test,y_test)

print(f"Naive Bayes Accuracy : ", nb_acc1)
print(f"Random Forest Accuracy : ", rf_acc)
print(f"SVM Accuracy : ", svm_acc)
