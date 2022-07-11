import pandas as pd
import numpy as np
#import seaborn as sns
#import os





def preprocess(data):
    data = pd.DataFrame(data)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(data.mean().round(1), inplace=True)

    # Feature set

    feature_set = data.columns.to_list()

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    prod_x = data[feature_set]
    scaler.fit(prod_x)

    prod_x = scaler.transform(prod_x)
    return prod_x
