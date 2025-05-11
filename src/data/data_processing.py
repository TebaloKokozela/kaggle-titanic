
from src.data.data_loader import load_data
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd


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



def data_processor(df):
    # fillna on the Age variable
    df['Missing_Age'] = df['Age'].isnull()
    df.fillna({'Age':-1},inplace=True)


    # create a columns to indicate missing Cabin numbers
    df['Missing_Cabin'] = pd.isnull(df['Cabin'])

    # Map male and female to numbers (0,1)
    df['Sex'] = df['Sex'].map({'male':0,'female':1})

    # map embarked to (0,1,2)
    df['Embarked'] = df['Embarked'].map({'C':0,'Q':1,'S':'2'})
    df.fillna({'Embarked':0},inplace=True)

    # count number of passengers on each ticket
    ticket_capacity = df['Ticket'].value_counts()
    # quick rename to avoid possible conflicts or confusion
    ticket_capacity.name = 'members'
    df =  df.merge(right=ticket_capacity, left_on='Ticket', right_index=True)

    # remove columns that won't be used for model
    df.drop(['Name','Cabin','Ticket','Deck'], axis=1, inplace=True)


    return df

