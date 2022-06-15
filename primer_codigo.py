#  Relevant Information:
#
#      Samples arrive periodically as Dr. Wolberg reports his clinical cases.
#      The database therefore reflects this chronological grouping of the data.
#      This grouping information appears immediately below, having been removed
#      from the data itself:
#
#        Group 1: 367 instances (January 1989)
#        Group 2:  70 instances (October 1989)
#        Group 3:  31 instances (February 1990)
#        Group 4:  17 instances (April 1990)
#        Group 5:  48 instances (August 1990)
#        Group 6:  49 instances (Updated January 1991)
#        Group 7:  31 instances (June 1991)
#        Group 8:  86 instances (November 1991)
#        -----------------------------------------
#        Total:   699 points (as of the donated datbase on 15 July 1992)
#
#      Note that the results summarized above in Past Usage refer to a dataset
#      of size 369, while Group 1 has only 367 instances.  This is because it
#      originally contained 369 instances; 2 were removed.  The following
#      statements summarizes changes to the original Group 1's set of data:
#
#      #####  Group 1 : 367 points: 200B 167M (January 1989)
#      #####  Revised Jan 10, 1991: Replaced zero bare nuclei in 1080185 & 1187805
#      #####  Revised Nov 22,1991: Removed 765878,4,5,9,7,10,10,10,3,8,1 no record
#      #####                  : Removed 484201,2,7,8,8,4,3,10,3,4,1 zero epithelial
#      #####                  : Changed 0 to 1 in field 6 of sample 1219406
#      #####                  : Changed 0 to 1 in field 8 of following sample:
#      #####                  : 1182404,2,3,1,1,1,2,0,1,1,1
#
#   5. Number of Instances: 699 (as of 15 July 1992)
#
#   6. Number of Attributes: 10 plus the class attribute
#
#   7. Attribute Information: (class attribute has been moved to last column)
#
#      #  Attribute                     Domain
#      -- -----------------------------------------
#      1. Sample code number            id number
#      2. Clump Thickness               1 - 10
#      3. Uniformity of Cell Size       1 - 10
#      4. Uniformity of Cell Shape      1 - 10
#      5. Marginal Adhesion             1 - 10
#      6. Single Epithelial Cell Size   1 - 10
#      7. Bare Nuclei                   1 - 10
#      8. Bland Chromatin               1 - 10
#      9. Normal Nucleoli               1 - 10
#     10. Mitoses                       1 - 10
#     11. Class:                        (2 for benign, 4 for malignant)
#
#   8. Missing attribute values: 16
#
#      There are 16 instances in Groups 1 to 6 that contain a single missing
#      (i.e., unavailable) attribute value, now denoted by "?".
#
#   9. Class distribution:
#
#      Benign: 458 (65.5%)
#      Malignant: 241 (34.5%)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as se

import matplotlib.pyplot as plot

plot.style.use('ggplot')

dataFrame = pd.read_csv('breast-cancer-wisconsin.csv')

print(dataFrame.columns)  # se tienen 11 columnas

# la columna bare_nucleoli tiene datos faltantes, se aplica preprocesamiento y se rellenan esos datos
print(dataFrame.describe(include='all').T)

df = pd.read_csv('breast-cancer-wisconsin-preprocesado.csv')

print(df.columns)  # se tienen 11 columnas

# la columna bare_nucleoli tiene datos faltantes, se aplica preprocesamiento y se rellenan esos datos
print(df.describe(include='all').T)

plot.figure(figsize=(8, 7))
se.countplot(x="class", data=df)
plot.title('Diagnosticos')
plot.show()

scaler = StandardScaler()
scaler.fit(df)

scaled_df = scaler.transform(df)

pca = PCA(n_components=2)
pca.fit(scaled_df)
x_pca = pca.transform(scaled_df)

pca_df = pd.DataFrame(data=x_pca, columns=['componente 1', 'componente 2'])
print(pca_df)

# plot de los componentes principales

plot.figure(figsize=(16, 8))
plot.xticks(fontsize=12)
plot.yticks(fontsize=12)

plot.xlabel('Componenete 1', fontsize=20)
plot.ylabel('Componenete 2', fontsize=20)

plot.title('Componentes principales de Cancer de Mama', fontsize=20)
targets = [0, 1]
colors = ['r', 'g']
for t, cl in zip(targets, colors):
    i = df['class'] == t
    print(i)
    plot.scatter(pca_df.loc[i, 'componente 1'],
                 pca_df.loc[i, 'componente 2'], c=cl, s=50)
plot.legend([2, 4], prop={'size': 15})


# confiabilidad, matriz de confusion, 80-20

X = df.drop('class', axis=1)
y = df['class']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('training', X_train.shape)
print('test', Y_train.shape)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(X_train, Y_train)

Y_LR = LR.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

plot.figure(figsize=(5, 2))
plot.title('Matriz de confusion', fontsize=16)
se.heatmap(confusion_matrix(Y_test, Y_LR), annot=True, cmap='viridis', fmt='.0f')
plot.show()

print(classification_report(Y_test, Y_LR))

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, Y_LR)
print('Confiabilidad es: ', accuracy)


# 50/50

X = df.drop('class', axis=1)
y = df['class']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.5, random_state=42)

print('training', X_train.shape)
print('test', Y_train.shape)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(X_train, Y_train)

Y_LR = LR.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

plot.figure(figsize=(5, 2))
plot.title('Matriz de confusion', fontsize=16)
se.heatmap(confusion_matrix(Y_test, Y_LR), annot=True, cmap='viridis', fmt='.0f')
plot.show()

print(classification_report(Y_test, Y_LR))

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, Y_LR)
print('Confiabilidad es: ', accuracy)