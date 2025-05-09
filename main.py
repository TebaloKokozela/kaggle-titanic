from src.data.data_loader import load_data
from src.data.data_processing import  data_processor,process_deck



df = load_data(file_path='data/raw/train.csv')

df, encoder =  process_deck(df,is_train=True,encoder=None)

df = data_processor(df)

df.to_csv('data/processed/titanic_clean.csv',index=True)
