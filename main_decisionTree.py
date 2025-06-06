from src.data.data_loader import load_data
from src.data.data_processing import  data_processor,process_deck
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score

print('Running file main.py')

df = load_data(file_path='data/raw/train.csv')
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
X = df.drop('Survived',axis=1)

print(X.columns == df_test.columns)

lr = DecisionTreeClassifier(max_depth=4)
lr.fit(X,Y)

print(f"{'='*100}")
print(f"Test Score: {roc_auc_score(Y,lr.predict(X))}" )
print(f"{'='*100}")

print(df_test[X.columns].isnull().sum())

y_pred =  lr.predict(df_test[X.columns])

# pd.DataFrame({'Survived':y_pred},
#              index=df_test.index).to_csv('data/submission/gender_submission_v4.csv',index=True)
