import pandas as pd
from matplotlib import colors as mcolors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from random import shuffle
import time

def data_load(train_directory, test_directory):
    
    traindf = pd.read_json(train_directory)
    testdf = pd.read_json(test_directory)
    
    return traindf, testdf

def lda_visualization(x_data, df):
    
    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
    shuffle(colors)
    colors_select =[colors[i] for i in list(range(1,156,8))]
    
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_r = lda.fit(x_data, df['cuisine']).transform(x_data)
    
    plt.figure(figsize=(10,7))
    lw = 2
    for color, target_name in zip(colors_select, list(set(df['cuisine']))):
        plt.scatter(X_r[list(df['cuisine'].loc[df['cuisine']==target_name].keys()), 0], X_r[list(df['cuisine'].\
                    loc[df['cuisine']==target_name].keys()), 1], color=color, alpha=.8, lw=lw,label=target_name)    
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('LDA of Cooking dataset')
    plt.show()
    
def save_submission(df, predictions):
    df['cuisine'] = predictions
    df = df[['id','cuisine']]
    df[['id','cuisine']].to_csv('../output/{}.xlsx'.format(time.strftime("%Y-%m-%d-%I:%M",time.localtime())),index=False)
    print("저장했다.")