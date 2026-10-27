import numpy as np
import pandas as pd

if __name__ == "__main__":
    # print(np.linspace(0,10,3))
    # print(np.linspace(10,10000000,10, dtype=int))
    dyof = pd.DataFrame([[1, 1, 'odd'], [2, 4, 'NaN'], [3, 9, 'odd'], [4, 16, 'even'],[2, 4, 'NaN'],[2, 4, 'NaN']], columns=['x', 'x^2', 'parity'])
    df = pd.DataFrame([[1.9,'y','n'],[4.0,'ns','ys']],columns=['x','bo','po'])
    ya_df = pd.DataFrame([[1, 1, 'odd'], [2, 4, 'even'], [3, 9, 'odd'], [4, 16, 'even']], columns=['x', 'x^2', 'parity'])

    print(df)
    print("next\n\n")
    del_col = []
    yo = df.select_dtypes(include='number')
    print(yo)
    # for idx, col in enumerate(df.dtypes):
    #     print(col)
    #     if (col != 'int64') and 'float64':
    #         del_col.append(df.columns[idx])
    # for n in del_col:
    #     df.drop(n, axis=1, inplace=True)
    print(df)