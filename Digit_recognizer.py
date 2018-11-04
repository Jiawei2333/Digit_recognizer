import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

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