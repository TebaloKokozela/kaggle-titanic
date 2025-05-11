from configs import features
from src.data.data_loader import  load_data
from sklearn.model_selection import train_test_split

def train_test_read(file_path="../../data/processed/titanic_clean.csv"):
    #global X_train, X_test, y_train, y_test
    df = load_data(file_path)
    print(features)
    y = df['Survived']
    X = df.drop('Survived',axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33,
                                                        random_state=42)
    return  X_train, X_test, y_train, y_test