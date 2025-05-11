import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np
from src.data.data_loader import load_data
from src.data.data_processing import  data_processor,process_deck
from sklearn.feature_selection import f_classif,SelectPercentile
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 100
sns.set_style('whitegrid')

df = load_data(file_path='../../data/processed/titanic_clean.csv')

y_val = df['Survived']
X_val = df.drop('Survived',axis=1)


f_statistics, p_value =  f_classif(X=X_val,y=y_val)

df_pval = pd.DataFrame({'f_statistics':f_statistics,'p_values':p_value,},index=X_val.columns)

print(y_val.value_counts())
mat = df.corr()

fig, axs =  plt.subplots(1,2,figsize=(10, 8))


# First subplot: heatmap
sns.heatmap(mat, center=0, annot=True, linewidths=0.5, cmap='coolwarm',ax=axs[0])
axs[0].set_title('Heatmap of Feature Correlation/Importance')

# Second subplot: horizontal bar plot
df_pval.plot(kind='barh', legend=False,ax = axs[1] )
axs[1].set_title('Feature P-values')
axs[1].set_xlabel('P-value')

plt.tight_layout()
plt.savefig('../../outputs/plots/feature_selection.png')
plt.show()


selectPerc =  SelectPercentile(f_classif,percentile=50)
X_new = selectPerc.fit_transform(X_val,y_val)


print(selectPerc.get_feature_names_out())
