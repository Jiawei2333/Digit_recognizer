import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict

df_train = pd.read_csv(os.path.join(os.getcwd(), 'kaggle/Digit_recognizer/train.csv'))
df_test = pd.read_csv(os.path.join(os.getcwd(), 'kaggle/Digit_recognizer/test.csv'))

y_train = df_train.pop('label')
X_train = df_train.values
X_test = df_test.values

# Train Multi Layer Perceptron
from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(hidden_layer_sizes = [100], random_state = 0)
nn.fit(X_train, y_train)

predictions = nn.predict(X_test)
print(predictions[:10])

scores = cross_val_score(nn, X_train, y_train, scoring='accuracy', cv=3)
print('Acurracy for NN MLPClassifier is: ', scores)

# Manual check first 10 digits
fig = plt.figure()
for i in range(10):
    fig.add_subplot(2, 5, i+1)
    X1 = X_test[i].reshape(28,28)
    plt.imshow(X1)
    plt.axis('off')

# Plot Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix

y_predict = cross_val_predict(nn, X_train, y_train, cv = 3)
print(classification_report(y_train, y_predict))
confusion = confusion_matrix(y_train, y_predict)
plt.matshow(confusion, cmap = plt.cm.gray)
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

row_sums = confusion.sum(axis = 1, keepdims = True)
norm_confusion = confusion/row_sums # normalize the confusion matrix to check the errors
np.fill_diagonal(norm_confusion, 0)
plt.matshow(norm_confusion, cmap = plt.cm.gray)
plt.colorbar()

# Add random noise to the digits
noise1 = np.random.randint(0, 100, (len(X_train), 784))
noise2 = np.random.randint(0, 100, (len(X_test), 784))
X_train_noise = X_train + noise1
X_test_noise = X_test + noise2
y_train_noise = X_train

fig_noise = plt.figure()
fig_noise.add_subplot(1,2,1)
plt.imshow(X_train_noise[5].reshape(28,28))
plt.axis('off')
fig_noise.add_subplot(1,2,2)
plt.imshow(X_train[5].reshape(28,28))
plt.axis('off')

# Digit Cleaner (Multi output classifier)
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_noise, y_train_noise)

clean_digit = knn_clf.predict([X_test_noise[3]])
fig_test = plt.figure()
fig_test.add_subplot(1,3,1)
plt.imshow(X_test[3].reshape(28,28)) # Original X_test
plt.axis('off')
fig_test.add_subplot(1,3,2)
plt.imshow(X_test_noise[3].reshape(28,28)) # Noisy X_test_noise
plt.axis('off')
fig_test.add_subplot(1,3,3)
plt.imshow(clean_digit.reshape(28,28)) # Cleaned (predicted) digit from KNN
plt.axis('off')

# SVM
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler

# Randomly shuffle the data
np.random.seed(42)
rnd_idx = np.random.permutation(np.shape(X_train)[0])
X_train = X_train[rnd_idx]
y_train = y_train[rnd_idx]

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))

# Train SVM
SVM_clf = SVC() # Use default rfb kernel
SVM_clf.fit(X_train_scaled[:10000], y_train[:10000])

predictions_svm = SVM_clf.predict(X_test_scaled[:10])
print(predictions_svm)

# Acurracy using Cross validation
scores_svm = cross_val_score(SVM_clf, X_train_scaled[:10000], y_train[:10000], scoring='accuracy', cv = 3)
print('Acurracy for rbf SVM is: ', scores_svm)


# Optimize hyperparameter using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

rnd_parameter = {'gamma': reciprocal(0.001, 0.1), 'C': uniform(1, 10)}
rnd_search_cv = RandomizedSearchCV(SVM_clf, rnd_parameter, n_iter = 10)
rnd_search_cv.fit(X_train_scaled[:1000], y_train[:1000])

print('Best estimator: ', rnd_search_cv.best_estimator_)
print('Best score: ', rnd_search_cv.best_score_)

from sklearn.metrics import accuracy_score

prediction_cv = cross_val_predict(rnd_search_cv.best_estimator_, X_train_scaled, y_train, cv = 3)
print('Best accuracy of rbf SVC: ', accuracy_score(y_train, prediction_cv))
# cross_val_score(rnd_search_cv.best_estimator_, X_train_scaled[:10000], y_train[:10000], cv = 3)