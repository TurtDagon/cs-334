import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from tqdm import tqdm
from joblib import Parallel, delayed



def generate_feature_vector_challenge(df):
    #impossible (-1 values or impossible age/height/weight)
    bad = df['Value'] < 0
    df.loc[bad, 'Value'] = np.nan
    df.loc[df['Variable'].isin(['Age', 'Height', 'Weight']) & (df['Value'] == 0), 'Value'] = np.nan

    feature_dict = {}   

    static = df[df['Time'] == '00:00']
    feature_dict.update(static.set_index('Variable')['Value'].to_dict())

    timeseries = df[df['Time'] != '00:00']
    means = timeseries.groupby('Variable')['Value'].mean() 
    feature_dict.update({f"mean_{var}": val for var, val in means.items()})

    categorical = { 
    'Gender':  [0, 1],
    'ICUType': [1, 2, 3, 4],
    'MechVent': [0,1],
    }

    #one-hot the categorical
    for var, cats in categorical.items():
        val = feature_dict.pop(var, np.nan)
        for c in cats:
            # if the category is unknown, keep NaN in every dummy (not 0), so missingness stays visible
            feature_dict[f"{var}_{c}"] = np.nan if pd.isna(val) else float(val == c)
    
    return feature_dict
def impute_missing_values_challenge(X):
    """
    instead of replace by mean
    replace accordingly to given data by the static variables (age, gender, height, weight, icutype)

    if too much of the initial data is gone, then simply remove from consideration
    
    """
    #row is admission/person
    #each column is variable/feature
    columns = list(X.shape[0])  # get names b4 converting into np
    X = np.array(X, dtype=float)
    idx = {name: i for i, name in enumerate(columns)}
    AGE, H, W = idx['Age'], idx['Height'], idx['Weight']

    gender_cols = [i for n, i in idx.items() if n.startswith('Gender_')]
    icu_cols    = [i for n, i in idx.items() if n.startswith('ICUType_')]
    static_cols = [AGE, H, W] + gender_cols + icu_cols

    #if age, gender, ICUtype nan, then remove
    bad = np.isnan(X[:, AGE]) | np.isnan(X[:, gender_cols]).any(axis=1) | np.isnan(X[:, icu_cols]).any(axis=1)
    keep = ~bad
    X = X[keep]
    X_filled = X.copy()

    comp = X[~np.isnan(X[:, static_cols]).any(axis=1)] # rows with 'complete' data (all 5 static vars present)
    

    #fill out height and weight acc to gender
    h_nan, w_nan = np.isnan(X[:, H]), np.isnan(X[:, W])
    for gcol in gender_cols:
        c = comp[comp[:, gcol] == 1]
        h_from_w = LinearRegression().fit(c[:,[W]], c[:,H])
        w_from_h = LinearRegression().fit(c[:,[H]], c[:,W])
        gend = X[:,gcol] == 1

        # height missing, weight present
        m1 = gend & h_nan & ~w_nan
        if m1.any():
            X_filled[m1,H] = h_from_w.predict(X[m1][:, [W]])
        # weight missing, height present
        m2 = gend & w_nan & ~h_nan
        if m2.any():
            X_filled[m2,W] = w_from_h.predict(X[m2][:, [H]])
        
        # h&w missing: fall back to the gender mean
        m3 = gend & h_nan & w_nan
        X_filled[m3, H] = c[:,H].mean()
        X_filled[m3, W] = c[:,W].mean()


    #basic means based on all static variable for variables w/out data
    for col in range(X.shape[1]):
        if col in static_cols:
            continue
        has_value = ~np.isnan(X[:, col]) #var exists
        if not has_value.any(): # never measured: nothing to learn from
            continue

        train_X = X_filled[has_value][:, static_cols] # static variables (inputs)
        train_y = X[has_value, col] # this col (target/label)

        if len(train_y) < 10:   # too little data: basic mean
            fill = train_y.mean()
        else:
            fill = LinearRegression().fit(train_X, train_y).predict(X_filled[:, static_cols])

        X_filled[:, col] = np.where(np.isnan(X[:, col]), fill, X[:, col])

    return X_filled


def normalize_feature_matrix_challenge(X):
    X = np.array(X, dtype=float)
    mins = X.min(axis=0)
    ranges = X.max(axis=0)-mins
    ranges[ranges == 0] = 1     #so can't be divided by 0      
    X = (X - mins) / ranges
    return X

def main():
    inputfile = ("test.csv")
    # df = pd.read_csv(inputfile)
    # print(df)

    df_labels = pd.read_csv(inputfile)
    IDs = df_labels['RecordID']
    raw_data = {}
    for i in tqdm(IDs, desc='Loading files from disk'):
        raw_data[i] = pd.read_csv('data/files/{}.csv'.format(i))

    features = Parallel(n_jobs=16)(delayed(generate_feature_vector_challenge)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    df_features = pd.DataFrame(features)
    feature_names = df_features.columns.tolist()
    X, y = df_features.values, df_labels['30-day_mortality'].values
    print(X)
    # X = impute_missing_values_challenge(X)
    # X = normalize_feature_matrix_challenge(X)
if __name__ == "__main__":
    main()
    