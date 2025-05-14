import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from src.data.data_loader import load_data
import seaborn as sns
import matplotlib as mpl
sns.set_style('whitegrid')
mpl.rcParams['figure.dpi'] = 100




df = load_data(file_path='../../data/processed/titanic_clean.csv')

y_val = df['Survived']
X_val = df.drop('Survived',axis=1)

X_train,X_test,y_train,y_test = train_test_split(X_val,y_val,random_state=42,test_size=0.1)

dt = DecisionTreeClassifier()

max_ = int(np.log2(X_train.shape[0]*(3/4)))

params =  {'max_depth':np.arange(2,max_)}


clf = GridSearchCV(estimator=dt,param_grid=params,cv=4,return_train_score=True,verbose=2)
clf.fit(X_train,y_train)

cv_results_cv = pd.DataFrame(clf.cv_results_)
print(cv_results_cv.columns)

y_tePd = clf.predict(X_test)
y_trPd = clf.predict(X_train)



plt.figure()
plt.errorbar(y=cv_results_cv['mean_train_score'],yerr=cv_results_cv['std_train_score'],x=np.arange(2,max_))
plt.errorbar(y=cv_results_cv['mean_test_score'],yerr=cv_results_cv['std_test_score'],x=np.arange(2,max_))
plt.tight_layout()
plt.savefig('../../outputs/performance_plots/DecisionTree.png')
plt.show()

print(f"\n {'='*100} \n")
print(f"Test roc_score : {roc_auc_score(y_test,y_tePd)}")
print(f"Train roc_score : {roc_auc_score(y_train,y_trPd)}")
print(f"\n {'='*100} \n")

print(clf.best_params_)
print(clf.best_score_)


