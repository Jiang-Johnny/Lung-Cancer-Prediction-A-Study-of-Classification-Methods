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
