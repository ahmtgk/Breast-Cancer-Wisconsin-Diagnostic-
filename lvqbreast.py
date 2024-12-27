# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:00:58 2024

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
from sklearn.model_selection import train_test_split

class LVQ:
    def __init__(self, learning_rate=0.8, epochs=50):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.unique_classes = np.unique(y)
        self.class_prototypes = {class_label: np.mean(X[y == class_label], axis=0) for class_label in self.unique_classes}

        for _ in range(self.epochs):
            for x, target in zip(X, y):
                closest_class = self.predict_class(x)
                if closest_class == target:
                    self.class_prototypes[closest_class] += self.learning_rate * (x - self.class_prototypes[closest_class])
                else:
                    self.class_prototypes[closest_class] -= self.learning_rate * (x - self.class_prototypes[closest_class])

    def predict_class(self, x):
        distances = [np.linalg.norm(x - prototype) for prototype in self.class_prototypes.values()]
        return list(self.class_prototypes.keys())[np.argmin(distances)]

    def predict(self, X):
        return [self.predict_class(x) for x in X]

if __name__ == "__main__":    
    

    veri = pd.read_csv("data.csv")
    veri =veri.iloc[:,:32]
    x=veri.iloc[:,2:]
    y=veri.iloc[:,1:2]

    from sklearn.preprocessing import LabelEncoder
    lbl = LabelEncoder()
    Y= lbl.fit_transform(y)

    X=x

    
    
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(Y)
    X = X_df.values
    y = y_df.values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
    
    # LVQ modelini eğitme
    lvq = LVQ(learning_rate=0.8, epochs=50)
    lvq.fit(X_train, y_train)

    # Tahmin yapma
    test_input = X_test  # Rastgele bir giriş örneği
    predicted_class = lvq.predict(test_input)
    
    #print("Tahmin edilen sınıf:", predicted_class)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
    from sklearn.metrics import confusion_matrix,precision_score,recall_score, ConfusionMatrixDisplay
    cm= confusion_matrix(y_test, predicted_class)
    print("lvq")
    print(cm)
    rapor = classification_report(y_test, predicted_class)
    print(rapor)
    # ROC eğrisi için gerekli olan false positive oranı (fpr) ve true positive oranı (tpr) değerlerini hesapla
    fpr, tpr, _ = roc_curve(y_test, predicted_class)

    # ROC eğrisini çiz
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--')  # Rastgele tahminin referans çizgisi
    plt.xlabel('False Positive Oranı')
    plt.ylabel('True Positive Oranı')
    plt.title('ROC Eğrisi lvq')
    plt.show()
    # accuracy(doğruluk)
    acc_score = accuracy_score(y_test, predicted_class)
    print("accuracy:", acc_score)

    # precision(kesinlik)
    prec_score = precision_score(y_test, predicted_class)
    print("precision:", prec_score)

    # recall(duyarlılık) -> TP/(TP+FN)
    rec_score = recall_score(y_test, predicted_class)
    print("recall:", rec_score)

    # specificity(özgüllük) -> TN/(TN+FP)
    tn = cm[1][1]
    fp = cm[0][1]
    spec_score = tn/(tn+fp)
    print("specificity:", spec_score)

    # f1 score(f1 ölçümü)
    f1_score = f1_score(y_test, predicted_class)
    print("f1 ölçümü:", f1_score)

    # auc
    auc_score = roc_auc_score(y_test, predicted_class)
    print("auc değeri:", auc_score)

    # kappa
    kappa_score = cohen_kappa_score(y_test, predicted_class)
    print("kappa değeri:", kappa_score)
    conf_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    conf_matrix_disp.plot()