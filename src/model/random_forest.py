import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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
print(f'X_train,X_test,y_train,y_test {X_train.shape,X_test.shape,y_train.shape,y_test.shape }')
rf = RandomForestClassifier()

max_ = int(np.log2(X_train.shape[0]*(3/4)))

params =  {'max_depth':np.arange(2,max_),'n_estimators':np.arange(10,110,10)}

clf = GridSearchCV(estimator=rf,param_grid=params,cv=4,return_train_score=True,verbose=2)
clf.fit(X_train,y_train)

cv_results_cv = pd.DataFrame(clf.cv_results_)
print(cv_results_cv.columns)
print(cv_results_cv.shape)

y_tePd = clf.predict(X_test)
y_trPd = clf.predict(X_train)

#cv_results_cv['mean_train_score'].reshape(7,10)

df_train_cv_mean =  cv_results_cv.pivot_table(index='param_n_estimators',columns='param_max_depth',values='mean_train_score',aggfunc='sum')
df_test_cv_mean =  cv_results_cv.pivot_table(index='param_n_estimators',columns='param_max_depth',values='mean_test_score',aggfunc='sum')

df_train_cv_std =  cv_results_cv.pivot_table(index='param_n_estimators',columns='param_max_depth',values='std_train_score',aggfunc='sum')
df_test_cv_std =  cv_results_cv.pivot_table(index='param_n_estimators',columns='param_max_depth',values='std_test_score',aggfunc='sum')

df_train_cv_mean = df_train_cv_mean.T
df_train_cv_std = df_train_cv_std.T

df_test_cv_mean = df_test_cv_mean.T
df_test_cv_std = df_test_cv_std.T

print(df_train_cv_mean)
print(f"{'='*100}")
print(df_train_cv_mean.shape)



# fig, axes = plt.subplot(nrows=3,ncols=3)
# plt.errorbar(y=df_train_cv_mean['mean_train_score'],yerr=cv_results_cv['std_train_score'],x=np.arange(10,110,10),ax=axes[1])
# plt.errorbar(y=cv_results_cv['mean_test_score'],yerr=cv_results_cv['std_test_score'],x=np.arange(10,110,10),ax=axes[2])
# plt.tight_layout()
# plt.savefig('../../outputs/performance_plots/RandomForest.png')
# plt.show()
#
# print(f"\n {'='*100} \n")
# print(f"Test roc_score : {roc_auc_score(y_test,y_tePd)}")
# print(f"Train roc_score : {roc_auc_score(y_train,y_trPd)}")
# print(f"\n {'='*100} \n")
#
print(clf.best_params_)
print(clf.best_score_)
# print(cv_results_cv['params'].value_counts())
