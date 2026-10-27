gend = s[:,1] ==g
train = gend & ~np.isnan(comp[:, H]) & ~np.isnan(comp[:, W])
potential_h = LinearRegression().fit(comp[train][:, [W]], comp[train, H])
potential_w = LinearRegression().fit(comp[train][:, [H]], comp[train, W])

mean_from_gend = LinearRegression().fit()
# height missing, weight present
m1 = gend & np.isnan(X[:, H]) & ~np.isnan(X[:, W])
if m1.any():
    X_filled[m1, H] = potential_h.predict(X[m1][:, [W]])

# weight missing, height present
m2 = gend & np.isnan(X[:, W]) & ~np.isnan(X[:, H])
if m2.any():
    X_filled[m2, W] = potential_w.predict(X[m2][:, [H]])

#both missing
m3 = gend & np.isnan(X[:, W]) & np.isnan(X[:, H])
if m2.any():
    X_filled[m2, W] = meanfromgend





def generate_feature_vector(df):
    static_variables = config['static']
    timeseries_variables = config['timeseries']
    feature_dict = {}

    for var in static_variables:
        vals = df[df['Variable'] == var]['Value']
        v = vals.iloc[0] if len(vals) else np.nan
        feature_dict[var] = np.nan if v == -1 else v

    for var in timeseries_variables:
        vals = df[df['Variable'] == var]['Value']
        feature_dict[f'mean_{var}'] = vals.mean() if len(vals) else np.nan

    return feature_dict