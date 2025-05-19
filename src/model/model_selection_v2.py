import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns
import pandas as pd
import numpy as np
from src.data.data_loader import load_data
import matplotlib as mpl
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from src.model.model import X_train, X_test

mpl.rcParams['figure.dpi'] = 100
sns.set_style('whitegrid')


df =  load_data('../../data/processed/train_interactive.csv')
print(df.info())

Y = df['Survived']
X = df.drop('Survived',axis=1)

scalar =  StandardScaler()
X =  scalar.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=3)

print(Y.value_counts(dropna=False))
svm = SVC(gamma='scale')


step_points =  np.logspace(-3,3,7)

C_vals = step_points
gamma_vals = step_points

kernel_vals = ['rbf']

params_dict = {'kernel':kernel_vals,'C':C_vals,'gamma':gamma_vals}

gsi = GridSearchCV(estimator=svm,cv=4,scoring='roc_auc',
                   return_train_score=True,param_grid=params_dict,verbose=3)

gsi.fit(X_train,y_train)

results_df =  pd.DataFrame(gsi.cv_results_)

print(results_df.head())
print(results_df.columns)

print(f" shape = {results_df['mean_train_score'].shape}")


print(gsi.best_params_)
print(gsi.best_estimator_)
print(gsi.best_score_)


y_pred = gsi.predict(X_test)
y_train_roc = gsi.predict(X_train)

print(f"\n{'='*100}\n")
print(f"test roc: {roc_auc_score(y_test,y_pred)} ")
print(f"train roc: {roc_auc_score(y_train,y_train_roc)} ")
print(f"\n{'='*100}\n")


#
# plt.figure()
# plt.errorbar(y=results_df['mean_train_score'],yerr=results_df['std_train_score'],x=k_values,label='train score')
# plt.errorbar(y=results_df['mean_test_score'],yerr=results_df['std_test_score'],x=k_values,label='test score')
# plt.xticks(k_values)
# plt.title('Validation Curve for Knn$')
# plt.legend()
# plt.xlabel('K-values')
# plt.ylabel('roc auc score')
# plt.savefig('../../outputs/performance_plots/Validation_Curve_knn.png')
# plt.show()
#
