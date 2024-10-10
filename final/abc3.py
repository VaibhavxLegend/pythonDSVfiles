import pandas as pd
from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo

rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 
X = rice_cammeo_and_osmancik.data.features 
y = rice_cammeo_and_osmancik.data.targets 
df = X.copy()
df['Target'] = y

def pie_chart(df):
    features = ['Area', 'Perimeter', 'Major_Axis_Length']
    for feature in features:
        sums_by_target = df.groupby('Target').sum()

        sums_by_target[feature].plot(kind='pie', fontsize=20)
        plt.ylabel(feature, horizontalalignment='left')
        plt.title('Breakdown for ' + feature, fontsize=25)

        plt.savefig(f'rice_pie_for_{feature}.jpg')
        plt.close()

def bar_chart(df):
    sums_by_Target = df.groupby('Target').sum()
    var = 'Area'
    sums_by_Target[var].plot(kind='bar', fontsize=15, rot=30)
    plt.title('Breakdown for ' + var, fontsize=20)
    plt.savefig('rice_bar_for_one_variable.jpg')
    plt.close()

    sums_by_Target.plot(kind='bar', subplots=True, fontsize=12)
    plt.suptitle('Total Measurements, by Target')
    plt.savefig('rice_bar_for_each_variable.jpg')
    plt.close()

def histogram(df):
    df.drop(columns='Target').plot(kind='hist', subplots=True, layout=(3, 3), bins=20, figsize=(10, 8))
    plt.suptitle('Rice Histograms', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  
    plt.show()

def mean_mediam_mode(df):
    col = df['Area']
    Average = col.mean()
    Std = col.std()
    Median = col.median() 
    Perc25 = col.quantile(0.25)
    Perc75 = col.quantile(0.75)
    Clean_Avg = col[(col > Perc25) & (col < Perc75)].mean()

    print(f"Average: {Average}")
    print(f"Standard Deviation: {Std}")
    print(f"Median: {Median}")
    print(f"25th Percentile: {Perc25}")
    print(f"75th Percentile: {Perc75}")
    print(f"Clean Average (excluding outliers): {Clean_Avg}")

def box_plot(df):
    col = 'Area'
    df['ind'] = pd.Series(df.index)
    pivot_df = df.pivot(index='ind', columns='Target', values=col)
    pivot_df.plot(kind='box')
    plt.show()

def scatter_plot(df):
    df.plot(kind="scatter",
    x="Major_Axis_Length", y="Minor_Axis_Length")
    plt.title("Length vs Width")
    plt.show()
scatter_plot(df)