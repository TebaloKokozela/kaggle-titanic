import pandas as pd
import numpy as np
from src.data.data_loader import load_data
from sklearn.preprocessing import OrdinalEncoder

#df = load_data('data/raw/train.csv')





features = ['Pclass', 'Sex','Fare','Missing_Cabin','DeckNum']