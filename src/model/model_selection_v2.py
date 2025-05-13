import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import pandas as pd
import numpy as np
from src.data.data_loader import load_data
import matplotlib as mpl
from sklearn.metrics import roc_auc_score

from src.model.model import X_train, X_test

mpl.rcParams['figure.dpi'] = 100
sns.set_style('whitegrid')


df =  load_data('../../data/processed/train_interactive.csv')
print(df.info())

Y = df['Survived']
X = df.drop('Survived',axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.1,random_state=42)

print(Y.value_counts(dropna=False))
lri = LogisticRegression(penalty='l1',solver='liblinear')

c_vals = (10.0)**(np.arange(-3,3,0.5)*-1)

params_dict = {'C':c_vals}

gsi = GridSearchCV(estimator=lri,cv=4,scoring='roc_auc',
                   return_train_score=True,param_grid=params_dict,verbose=3)
gsi.fit(X_train,y_train)

results_df =  pd.DataFrame(gsi.cv_results_)

print(results_df.head())
print(results_df.columns)

print(f" shape = {results_df['mean_train_score'].shape}")

plt.figure()
plt.errorbar(y=results_df['mean_train_score'],yerr=results_df['std_train_score'],x=np.log10(c_vals),label='train score')
plt.errorbar(y=results_df['mean_test_score'],yerr=results_df['std_test_score'],x=np.log10(c_vals),label='test score')
plt.xticks(np.log10(c_vals))
plt.title('Train Test AUC Scores on different $log_{10}C$')
plt.legend()
plt.xlabel('$log_{10} (C)$')
plt.ylabel('roc auc score')
plt.savefig('../../outputs/performance_plots/logistic_GridSearch.png')
plt.show()


print(gsi.best_params_)
print(gsi.best_estimator_)
print(gsi.best_score_)


y_pred = gsi.predict(X_test)
y_train_roc = gsi.predict(X_train)

print(f"\n{'='*100}\n")
print(f"test roc: {roc_auc_score(y_test,y_pred)} ")
print(f"train roc: {roc_auc_score(y_train,y_train_roc)} ")
print(f"\n{'='*100}\n")

