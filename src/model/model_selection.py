from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from src.model.utils import train_test_read
import numpy as np

X_train,X_test,y_train,y_test = train_test_read()

lr = LogisticRegression()
np.set_printoptions(suppress=True)
print(np.arange(-3,3,0.5))
c_vals = 10.0**(np.arange(-3,3,0.5)*-1)
print(c_vals)
params = {'C':c_vals,'penalty':['l1','l2'],'solver':['liblinear']}
clf = GridSearchCV(estimator=lr,param_grid=params,verbose=1,scoring='roc_auc',cv=4)

clf.fit(X_train,y_train)

print(clf.best_params_)
print(clf.best_score_)
print('\n')
print(f"Mean test score: {clf.cv_results_['mean_test_score']} \n")
print(f"std test score: {clf.cv_results_['std_test_score']}\n")
print(f"Rank test score: {clf.cv_results_['rank_test_score']} \m")