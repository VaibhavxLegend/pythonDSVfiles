import pandas as pd
from ucimlrepo import fetch_ucirepo

rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 
X = rice_cammeo_and_osmancik.data.features 
y = rice_cammeo_and_osmancik.data.targets 

df = pd.DataFrame(X)
df['Target'] = y  

def statistical_analysis(dataframe, feature):
    analysis = {}
    analysis['Mean'] = dataframe[feature].mean()
    analysis['Variance'] = dataframe[feature].var()
    analysis['Standard Deviation'] = dataframe[feature].std()
    analysis['Median'] = dataframe[feature].median()
    analysis['Mode'] = dataframe[feature].mode()[0] 
    return analysis

feature_to_analyze = 'Area' 
stats = statistical_analysis(df, feature_to_analyze)
print(f'Statistical Analysis for {feature_to_analyze}:')
for stat, value in stats.items():
    print(f'{stat}: {value}')