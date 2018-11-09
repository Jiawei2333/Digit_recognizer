import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neural_network import MLPClassifier

df_train = pd.read_csv(os.path.join(os.getcwd(), 'kaggle/Digit_recognizer/train.csv'))
df_test = pd.read_csv(os.path.join(os.getcwd(), 'kaggle/Digit_recognizer/test.csv'))

y_train = df_train.pop('label')
X_train = df_train.values
X_test = df_test.values

nn = MLPClassifier(hidden_layer_sizes = [100], random_state = 0)
nn.fit(X_train, y_train)

predictions = nn.predict(X_test)
print(predictions[:10])

scores = cross_val_score(nn, X_train, y_train, scoring = 'accuracy', cv = 3)
print('Acurracy for NN MLPClassifier is: ', scores)

# Check the first 10 instances by visualizing the image
fig = plt.figure()
for i in range(10):
    fig.add_subplot(2, 5, i+1)
    X1 = X_test[i].reshape(28,28)
    plt.imshow(X1)
	plt.axis('off')

# Plot confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

y_predict = cross_val_predict(nn, X_train, y_train, cv = 3)
print(classification_report(y_train, y_predict))
confusion = confusion_matrix(y_train, y_predict)
plt.matshow(confusion)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# SVM
from sklearn.svm import SVC

SVM_clf = SVC(kernel = 'linear', gamma = 0.1, C = 10, random_state = 0)
SVM_clf.fit(X_train, y_train)

predictions_svm = SVM_clf.predict(X_test)
print(predictions_svm[:10])

scores_svm = cross_val_score(SVM_clf, X_train, y_train, scoring='accuracy', cv=2)
print('Acurracy for rbf SVM is: ', scores_svm)