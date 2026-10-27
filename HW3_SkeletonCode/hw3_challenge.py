# hw3_challenge.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.linear_model import LinearRegression, LogisticRegression
import helper
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, exceptions
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

def generate_feature_vector_challenge(df):
    df = df.copy()
    #impossible (-1 values or impossible age/height/weight)
    df.loc[df['Value'] < 0, 'Value'] = np.nan
    df.loc[df['Variable'].isin(['Age', 'Height', 'Weight']) & (df['Value'] == 0), 'Value'] = np.nan

    static_vars = ['Age', 'Gender', 'Height', 'ICUType', 'Weight']
    feature_dict = {}

    static = df[(df['Time'] == '00:00')]
    feature_dict.update(static.set_index('Variable')['Value'].to_dict())

    ts = df[~df['Variable'].isin(static_vars + ['RecordID', 'MechVent'])].dropna(subset=['Value'])
    ts = ts.sort_values('Time')
    stats = ts.groupby('Variable')['Value'].agg(['mean', 'min', 'max', 'std', 'first', 'last', 'count'])
    for var, row in stats.iterrows():
        for stat, v in row.items():
            feature_dict[f"{stat}_{var}"] = v

    mech = df.loc[df['Variable'] == 'MechVent', 'Value']
    feature_dict['MechVent'] = mech.max() if mech.notna().any() else np.nan
    
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

def impute_missing_values_challenge(X, columns):
    """
    instead of replace by mean
    replace accordingly to given data by the static variables (age, gender, height, weight, icutype)

    if too much of the initial data is gone, then simply remove from consideration
    
    """
    #row is admission/person
    #each column is variable/feature
    X = np.array(X, dtype=float)
    X_filled = X.copy()


    columns = [str(c) for c in columns] #not getting read as string
    idx = {name: i for i, name in enumerate(columns)}
    #returns [('Age', 0), ('Height', 1), ('Weight', 2), ('mean_BUN', 3), ('mean_Creatinine', 4), ('mean_GCS', 5), ('mean_Glucose', 6), ('mean_HCO3', 7), ('mean_HCT', 8), ('mean_HR', 9)
    AGE, H, W = idx['Age'], idx['Height'], idx['Weight']
    
    gender_cols = [i for n, i in idx.items() if n.startswith('Gender_')]
    icu_cols    = [i for n, i in idx.items() if n.startswith('ICUType_')]
    static_cols = [AGE, H, W] + gender_cols + icu_cols

    #if age, gender, ICUtype nan, then remove (for training)
    bad = np.isnan(X[:, AGE]) | np.isnan(X[:, gender_cols]).any(axis=1) | np.isnan(X[:, icu_cols]).any(axis=1)
    keep = ~bad
    #need same rows
    X_filled[np.isnan(X[:, AGE]), AGE] = np.nanmean(X[keep, AGE])
    

    for group in (gender_cols, icu_cols):
        group = np.array(group)
        common = group[np.argmax(X[keep][:, group].mean(axis=0))]# most common category
        rows = np.isnan(X[:, group]).any(axis=1)# rows with unknown category
        X_filled[np.ix_(rows, group)] = 0.0
        X_filled[rows, common] = 1.0 
    
    
    comp = X[~np.isnan(X[:, static_cols]).any(axis=1)] # rows with 'complete' data (all 5 static vars present)
    #fill out height and weight acc to gender
    h_nan, w_nan = np.isnan(X[:, H]), np.isnan(X[:, W])
    for gcol in gender_cols:
        c = comp[comp[:, gcol] == 1]
        h_from_w = LinearRegression().fit(c[:,[W]], c[:,H])
        w_from_h = LinearRegression().fit(c[:,[H]], c[:,W])
        gend = X_filled[:, gcol] == 1

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
        X_filled[:, col] = np.where(has_value, X[:, col], fill)
    X_filled = np.nan_to_num(X_filled, nan=0.0)
    return X_filled

def normalize_feature_matrix_challenge(X):
    X = np.array(X, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        if len(np.unique(col)) > 2:                 # skip one-hot/binary columns
            lo, hi = np.percentile(col, [1, 99])
            X[:, j] = np.clip(col, lo, hi)
    mins = X.min(axis=0)
    ranges = X.max(axis=0) - mins
    ranges[ranges == 0] = 1
    return (X - mins) / ranges

def conf_matrix(clf, X,y_true,y_pred, y_score):
    cm = metrics.confusion_matrix(y_true, y_pred, labels=[-1,1])
    tn = cm[0,0]
    fn = cm[1,0]
    fp = cm[0,1]
    tp = cm[1,1]
    f1 = metrics.f1_score(y_true,y_pred)
    auc = metrics.roc_auc_score(y_true,y_score)
    print(f"TN: {tn}\nFN: {fn}\nFP: {fp}\nTP: {tp}")
    print(f"f1score: {f1}\nAUROC: {auc}")

def get_challenge_data():
    """
    This function is similar to helper.get_train_test_split, except that:
    - It reads in all 10,000 training examples
    - It does not return labels for the 2,000 examples in the heldout test set
    You should replace your preprocessing functions (generate_feature_vector,
    impute_missing_values, normalize_feature_matrix) with updated versions for the challenge
    """
    df_labels = pd.read_csv('data/labels.csv')
    df_labels = df_labels
    IDs = df_labels['RecordID']
    raw_data = {}
    for i in tqdm(IDs, desc='Loading files from disk'):
        raw_data[i] = pd.read_csv('data/files/{}.csv'.format(i))

    features = Parallel(n_jobs=16)(delayed(generate_feature_vector_challenge)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    df_features = pd.DataFrame(features)
    feature_names = df_features.columns.tolist()
    X, y = df_features.values, df_labels['30-day_mortality'].values
    X = impute_missing_values_challenge(X, feature_names)
    X = normalize_feature_matrix_challenge(X)
    return X[:10000], y[:10000], X[10000:], feature_names


def run_challenge(X_challenge, y_challenge, X_heldout):
    # TODO:
    # Read challenge data
    # Train a linear classifier and apply to heldout dataset features
    # Use make_challenge_submission to print the predicted labels
    print("================= Part 3 ===================")
    print("Part 3: Challenge")
    clf = LogisticRegression(penalty='elasticnet',solver='saga',l1_ratio=0.5,max_iter=1000, class_weight='balanced') ### TODO: define your classifier with appropriate hyperparameters
    """
        testing & for confusion matrix
    """
    # X_train, X_test, y_train, y_test = train_test_split(X_challenge, y_challenge, test_size=2000)
    # thresholds = np.arange(0.05, 0.6, 0.01)
    # f1s = [f1_score(y_val, np.where(val_score > t, 1, -1), pos_label=1) for t in thresholds]
    # best_t = thresholds[int(np.argmax(f1s))]
    # print(best_t, max(f1s))

    #original codw
    clf.fit(X_challenge, y_challenge)
    threshold = 0.59

    #conf matrix
    train_score = clf.predict_proba(X_challenge)[:, 1]
    y_pred_train = np.where(train_score>threshold, 1, -1)
    conf_matrix(clf, X_challenge, y_challenge, y_pred_train, train_score)

    y_score = clf.predict_proba(X_heldout)[:, 1]
    y_label = np.where(y_score > threshold, 1, -1)

    helper.make_challenge_submission(y_label, y_score)
    

    

if __name__ == '__main__':
    # Read challenge data
    X_challenge, y_challenge, X_heldout, feature_names = get_challenge_data()

    # TODO: Question 3: Apply a classifier to heldout features, and then use
    #       make_challenge_submission to print the predicted labels

    run_challenge(X_challenge, y_challenge, X_heldout)
    helper.test_challenge_output()
