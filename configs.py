import pandas as pd
import numpy as np
from src.data.data_loader import load_data
from sklearn.preprocessing import OrdinalEncoder

df = load_data('data/raw/train.csv')


def process_deck(df,is_train = False, encoder = None):
    # handle missing data
    df['Missing_Cabin'] = pd.isnull(df['Cabin'])
    mask_cabin = df['Missing_Cabin'] == False
    df['Deck'] = (df.loc[mask_cabin, 'Cabin'].apply(lambda m: m[0]))
    # remove special case 'T'
    df['Deck'].replace('T', np.nan)
    # convert data to array
    array_form = np.array(df['Deck']).reshape(-1, 1)

    if is_train and encoder is None:
        encoder = OrdinalEncoder()
        encoder.fit(array_form)
        df['DeckNum'] = encoder.transform(array_form)
        df.fillna({'DeckNum':-1},inplace=True)

    else:
        encoder.transform(array_form)

    return df, encoder


features = ['Pclass' 'Sex' 'Fare' 'DeckNum']