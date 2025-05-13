import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np

from src.data.data_loader import load_data
from src.data.data_processing import  data_processor,process_deck
from sklearn.feature_selection import f_classif,SelectPercentile
import matplotlib as mpl
from sklearn.preprocessing import PolynomialFeatures

mpl.rcParams['figure.dpi'] = 100
sns.set_style('whitegrid')

df = load_data(file_path='../../data/processed/titanic_clean.csv')

y_val = df['Survived']
X_val = df.drop('Survived',axis=1)

print(y_val.value_counts())

f_statistics, p_value =  f_classif(X=X_val,y=y_val)

df_pval = pd.DataFrame({'f_statistics':f_statistics,'p_values':p_value,},index=X_val.columns)
df_pval.sort_values('f_statistics',ascending=False,inplace=True)
print(y_val.value_counts())
mat = df.corr()

# fig, axs =  plt.subplots(1,2,figsize=(10, 8))


# First subplot: heatmap
# sns.heatmap(mat, center=0, annot=True, linewidths=0.5, cmap='coolwarm',ax=axs[0])
# axs[0].set_title('Heatmap of Feature Correlation/Importance')
#
# # Second subplot: horizontal bar plot
# df_pval.plot(kind='barh', legend=False,ax = axs[1] )
# axs[1].set_title('Feature P-values')
# axs[1].set_xlabel('P-value')
#
# plt.tight_layout()
# plt.savefig('../../outputs/plots/feature_selection.png')
# plt.show()


selectPerc =  SelectPercentile(f_classif,percentile=50)
X_new = selectPerc.fit_transform(X_val,y_val)

features =  selectPerc.get_feature_names_out()

print(features)

# Feature Selection

poly_feat =  PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
X_inter =  poly_feat.fit_transform(X_val)


print(f"Number shape: {X_inter.shape} \n{X_val.shape} \nType: {type(X_inter)}")

feature_names =  poly_feat.get_feature_names_out(X_val.columns)

Poly_X = pd.DataFrame(X_inter)
Poly_X.columns = feature_names
print(Poly_X.head())
f_statistic, p_values = f_classif(X_inter, y_val)

# select top 20th percetile features
select_per =  SelectPercentile(score_func=f_classif,percentile=20)
select_per.fit_transform(Poly_X,y_val)
print(select_per.get_feature_names_out())

print(f"Poly_shape: {Poly_X.shape} y_val shape : {y_val.shape}")
Poly_X_filter = Poly_X #[select_per.get_feature_names_out()]
Poly_X_filter.index = X_val.index
Poly_X_filter.index.name = X_val.index.name
print('\n')
print(f"pd.concat : {pd.concat([Poly_X_filter,y_val],axis=1).shape} ")
print('\n')
print(f"Poly_X_filter: {Poly_X_filter.shape}")
print(Poly_X_filter.head())

new_df =  pd.concat([Poly_X_filter,y_val],axis=1)
new_df.to_csv('../../data/processed/train_interactive.csv',index=True)

# calculate f_statistics and it's corresponding P-values
df_fstats = pd.DataFrame({'F_statistics':f_statistic,'P_value':p_values},index = feature_names )
df_fstats.sort_values('F_statistics',ascending=False,inplace=True)


# fig,axs =  plt.subplots(nrows=1,ncols=2)
# df_fstats['F_statistics'].plot(kind='barh',ax=axs[0])
# axs[0].set_xlabel('F-Statistics')
# axs[0].set_title('F Statistics')
#
# df_fstats['P_value'].plot(kind='barh',ax=axs[1])
# axs[1].set_xlabel('P_value')
# axs[1].set_title('P_value')
#
# plt.tight_layout()
# plt.savefig('../../outputs/plots/fstat_pvalue.png')
# plt.show()








