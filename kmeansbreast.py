# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:16:45 2024

@author: Ahmet
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix,precision_score,recall_score, ConfusionMatrixDisplay
import scipy.stats as stats

#verimizi aldık id sütünunu attık sonuc stunundan itibaren aldık sonra değre sutununu y değerine atadık
veri = pd.read_csv("data.csv")
veri =veri.iloc[:,:32]
x=veri.iloc[:,2:]
y=veri.iloc[:,1:2]

#m ve b değerini 0 ve 1 e cevirdik
from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
Y= lbl.fit_transform(y)

X=x

# burada kaç küme olacağını sectik algoritma parametreleri
kmeans= cluster.KMeans(n_clusters=2,random_state=0)
kmeans.fit(X)
merkezler =kmeans.cluster_centers_.round(2)
tahmin = kmeans.labels_

cm= confusion_matrix(Y, tahmin)
print("kmeans")
print(cm)
rapor = classification_report(Y, tahmin)
print(rapor)
# ROC eğrisi için gerekli olan false positive oranı (fpr) ve true positive oranı (tpr) değerlerini hesapla
fpr, tpr, _ = roc_curve(Y, tahmin)

# ROC eğrisini çiz
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')  # Rastgele tahminin referans çizgisi
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC Eğrisi kmeans')
plt.show()
# accuracy(doğruluk)
acc_score = accuracy_score(Y, tahmin)
print("accuracy:", acc_score)

# precision(kesinlik)
prec_score = precision_score(Y, tahmin)
print("precision:", prec_score)

# recall(duyarlılık) -> TP/(TP+FN)
rec_score = recall_score(Y, tahmin)
print("recall:", rec_score)

# specificity(özgüllük) -> TN/(TN+FP)
tn = cm[1][1]
fp = cm[0][1]
spec_score = tn/(tn+fp)
print("specificity:", spec_score)

# f1 score(f1 ölçümü)
f1_score = f1_score(Y, tahmin)
print("f1 ölçümü:", f1_score)

# auc
auc_score = roc_auc_score(Y, tahmin)
print("auc değeri:", auc_score)

# kappa
kappa_score = cohen_kappa_score(Y, tahmin)
print("kappa değeri:", kappa_score)
conf_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=cm)
conf_matrix_disp.plot()
