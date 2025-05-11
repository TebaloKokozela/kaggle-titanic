
from src.model.utils import train_test_read
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

X_train, X_test, y_train, y_test =  train_test_read()

lr = LogisticRegression(verbose=1,C=10**0.5,penalty='l1',solver='liblinear')
lr.fit(X_train,y_train)


y_test_pred =  lr.predict(X_test)
y_tran_pred =  lr.predict(X_train)

#print(lr.__dict__)

roc_test  =  roc_auc_score(y_test,y_test_pred)
roc_train =  roc_auc_score(y_train,y_tran_pred)

print(f"Train ROC_AUC : {roc_train:0.4f} \nTest ROC_AUC : {roc_test:0.4f} \n")
np.set_printoptions(suppress=True)
print(f"{lr.coef_[0] }")

