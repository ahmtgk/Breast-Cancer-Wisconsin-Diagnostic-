# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:06:48 2024

@author: Ahmet
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

veri = pd.read_csv("data.csv")
veri =veri.iloc[:,:32]
x=veri.iloc[:,2:]
y=veri.iloc[:,1:2]

from sklearn.preprocessing import LabelEncoder
lbl = LabelEncoder()
Y= lbl.fit_transform(y)

X=x

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

lookback=1
model = Sequential()

model.add(SimpleRNN(units = 50,
                    activation = 'tanh',
                    input_shape=(X_train.shape[1],lookback)))
model.add(Dropout(0.2))
model.add(Dense(2))


model.compile(loss='mean_squared_error',optimizer='adam' )
callbacks = [EarlyStopping(monitor='val_loss',patience = 3,verbose = 1,mode= "min"),
             ModelCheckpoint(filepath='mymodel1.keras',monitor='val_loss',mode="min",
                             save_best_only=True,save_weights_only=False,verbose=1)
             ]
history = model.fit(x=X_train,
                    y=Y_train,
                    epochs=50,
                    batch_size=42,
                    validation_data=(X_test,Y_test),
                    callbacks=callbacks,
                    shuffle= False
    )

# Modeli değerlendir ve loss hesapla
loss = model.evaluate(X_test, Y_test, batch_size=1)

# Accuracy hesapla
predictions = model.predict(X_test)
rounded_predictions = [round(x[0]) for x in predictions]
correct_predictions = [1 if pred == true else 0 for pred, true in zip(rounded_predictions, Y_test)]
accuracy = sum(correct_predictions) / len(correct_predictions)
predictions = rounded_predictions
print("\nTest loss: %.4f" % loss)
print("Test accuracy: %.2f%%" % (100.0 * accuracy))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix,precision_score,recall_score, ConfusionMatrixDisplay
cm= confusion_matrix(Y_test, predictions)
print("rnn")
print(cm)
rapor = classification_report(Y_test, predictions)
print(rapor)
# ROC eğrisi için gerekli olan false positive oranı (fpr) ve true positive oranı (tpr) değerlerini hesapla
fpr, tpr, _ = roc_curve(Y_test, predictions)

# ROC eğrisini çiz
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')  # Rastgele tahminin referans çizgisi
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC Eğrisi rnn')
plt.show()
# accuracy(doğruluk)
acc_score = accuracy_score(Y_test, predictions)
print("accuracy:", acc_score)

# precision(kesinlik)
prec_score = precision_score(Y_test, predictions)
print("precision:", prec_score)

# recall(duyarlılık) -> TP/(TP+FN)
rec_score = recall_score(Y_test, predictions)
print("recall:", rec_score)

# specificity(özgüllük) -> TN/(TN+FP)
tn = cm[1][1]
fp = cm[0][1]
spec_score = tn/(tn+fp)
print("specificity:", spec_score)

# f1 score(f1 ölçümü)
f1_score = f1_score(Y_test, predictions)
print("f1 ölçümü:", f1_score)

# auc
auc_score = roc_auc_score(Y_test, predictions)
print("auc değeri:", auc_score)

# kappa
kappa_score = cohen_kappa_score(Y_test, predictions)
print("kappa değeri:", kappa_score)
conf_matrix_disp = ConfusionMatrixDisplay(confusion_matrix=cm)
conf_matrix_disp.plot()
