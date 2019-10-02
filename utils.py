import pandas as pd
from sklearn import preprocessing

def load_dataset():
    headers = ["elevation", "aspect", "slope", "horizontal_distance_to_hydrology", "vertical_distance_to_hydrology", "hillshade_9am", "hillshade_noon", "hillshade_3pm", "horizontal_distance_to_fire_points"] + ["wilderness_area_{}".format(i) for i in range(4)] + ["soil_type_{}".format(i) for i in range(40)] + ["cover_type"]
    df = pd.read_csv("covtype.data", header = None, names = headers)
    df_X, df_Y = split_data(df)
    x = df_X.values
    min_max_scaler = preprocessing.StandardScaler()#MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df.iloc[:, :-1] = x_scaled
    df.iloc[:, -1] = df_Y
    df = balance_classes(df)
    print("Num samples: {} (i.e. {} samples per class)".format(df.shape[0], int(df.shape[0]/7.0)))
    print(df.columns)
    return df

def split_data(df):
    return df.iloc[:, :-1], df.iloc[:, -1]

def balance_classes(df):
    g = df.groupby('cover_type')
    g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
    return g
