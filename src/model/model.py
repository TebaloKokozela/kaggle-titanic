

from configs import features
from sklearn.linear_model import LogisticRegression
from src.data.data_loader import  load_data
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

df = load_data(file_path="../../data/processed/titanic_clean.csv")

print(features)

y = df['Survived']
X =  df[features]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)

lr = LogisticRegression(verbose=1)
lr.fit(X_train,y_train)


y_test_pred =  lr.predict(X_test)
y_tran_pred =  lr.predict(X_train)

print(lr.__dict__)

roc_test  =  roc_auc_score(y_test,y_test_pred)
roc_train =  roc_auc_score(y_train,y_tran_pred)

print(f"Train ROC_AUC : {roc_train:0.4f} \nTest ROC_AUC : {roc_test:0.4f} \n")

print(f"{lr.coef_}")