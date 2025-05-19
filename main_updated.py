import numpy as np

from src.data.data_loader import load_data
from src.data.data_processing import  data_processor,process_deck
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

print('Running file main.py')

df = load_data(file_path='data/raw/train.csv')
df_cols = load_data(file_path='data/processed/train_interactive.csv')


print(f'df {type(df)} after loading data')
df_test = load_data(file_path='data/raw/test.csv')
print(f'df {type(df_test)}')

df, encoder =  process_deck(df,is_train=True,encoder=None)
print(f'df {type(df)} after processing deck')
df_test, encoder = process_deck(df_test,is_train=False,encoder=encoder)
print(f'df {type(df_test)}')

df = data_processor(df,encoder=encoder)
print(f'df {type(df)} after data processing')
df_test = data_processor(df_test,encoder=encoder)
print(f'df {type(df_test)}')

print('\n')
print(df.columns)
print(df_test.columns)
print('\n')
# df.to_csv('data/processed/titanic_clean.csv',index=True)
# df_test.to_csv()

Y = df['Survived']
df_cols.drop('Survived',axis=1,inplace=True)
X = df.drop('Survived',axis=1)

Poly = PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
X_new = Poly.fit_transform(X)
X_test_new = Poly.fit_transform(df_test)

X_new = pd.DataFrame(X_new)
X_test_new = pd.DataFrame(X_test_new)
X_new.columns =  Poly.get_feature_names_out()
X_test_new.columns =  Poly.get_feature_names_out()

X_new.index = X.index
X_new.index.name = X.index.name

X = X_new[df_cols.columns]

X_test_new = pd.DataFrame(X_test_new)
X_test_new.index = df_test.index
X_test_new.index.name = df_test.index.name

df_test = X_test_new[df_cols.columns]

print(X.shape, df_cols.columns )
print(X.columns == df_test.columns)

lr = KNeighborsClassifier(n_neighbors=9)
lr.fit(X,Y)

print(f"\n{'='*100}\n")
print(roc_auc_score(Y,lr.predict(X)))
print(f"\n{'='*100}\n")

y_pred =  lr.predict(df_test)



pd.DataFrame({'Survived':y_pred},
             index=df_test.index).to_csv('data/submission/gender_submission_knn.csv',index=True)
