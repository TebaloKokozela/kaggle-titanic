
from src.data.data_loader import load_data
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from configs import features,process_deck

import pandas as pd


df, encoder = process_deck("../../data/raw/train.csv")



def data_processor(df,encoder = encoder):
    # fillna on the Age variable
    df.fillna({'Age':-1},inplace=True)

    # Working on the Cabin variable
    # create a columns to indicate missing Cabin numbers
    df['Missing_Cabin'] = pd.isnull(df['Cabin'])
    mask_cabin = df['Missing_Cabin'] == False
    df['Deck'] = (df.loc[mask_cabin, 'Cabin'].apply(lambda m: m[0]))
    df['Deck'].replace('T', np.nan)
    array_form = np.array(df['Deck']).reshape(-1, 1)
    df['DeckNum'] = encoder.transform(array_form)
    df.fillna({'DeckNum': -1}, inplace=True)

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
    df.drop(['Name','Cabin','Ticket','Deck','Missing_Cabin'], axis=1, inplace=True)


    return df

