"""
Pandas DataFrame Manipulation with Palmer Penguins Dataset
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def load_csv(inputfile):
    """
    Load the csv as a pandas data frame
    
    Parameters
    ----------
    inputfile : string
        filename of the csv to load

    Returns
    -------
    csvdf : pandas.DataFrame
        return the pandas dataframe with the contents
        from the csv inputfile
    """
    # TODO: Implement this function
    return pd.read_csv(inputfile)


def remove_na(inputdf, colname):
    """
    Remove the rows in the dataframe with NA as values 
    in the column specified.
    
    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe
    colname : string
        Name of the column to check and remove rows with NA

    Returns
    -------
    outputdf : pandas.DataFrame
        return the pandas dataframe with the modified contents
    """
    # TODO: Implement this function
    #inputdf.drop(inputdf[inputdf[colname]== "NaN"].index)
    return inputdf.dropna(subset=[colname])


def onehot(inputdf, colname):
    """
    Convert the column in the dataframe into a one hot encoding.
    The newly converted columns should be at the end of the data
    frame and you should also drop the original column.
    
    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe
    colname : string
        Name of the column to one-hot encode

    Returns
    -------
    outputdf : pandas.DataFrame
        return the pandas dataframe with the modified contents
    """
    # TODO: Implement this function
    tmp = pd.get_dummies(inputdf[colname],dtype=int)
    newdf = pd.concat([inputdf, tmp], axis=1)
    newdf.drop(colname, axis=1, inplace=True)
    return newdf


def to_numeric(inputdf):
    """
    Extract all the
    
    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe

    Returns
    -------
    outputnp : numpy.ndarray
        return the numeric contents of the input dataframe as a 
        numpy array
    """
    # TODO: Implement this function
    #remove non int values??
    #check for non int types
    
    #yikes not that
    # del_col = []
    # for idx, col in enumerate(inputdf.dtypes):
    #     if col != 'int64':
    #         del_col.append(inputdf.columns[idx])
    # for n in del_col:
    #     inputdf.drop(n, axis=1, inplace=True)
    newdf = inputdf.select_dtypes(include='number')
    return newdf.to_numpy()


def main():
    
    """
    boxplot
    """
    # df = remove_na(df, "bill_length_mm")
    # df = remove_na(df, "bill_depth_mm")
    # df = remove_na(df, "flipper_length_mm")
    # df = remove_na(df, "body_mass_g")

    # pengi = ['Adelie','Chinstrap','Gentoo']
    # yup  = []
    # for peggg in pengi:
    #     ad = df[df['species']==peggg]["bill_length_mm"]
    #     yup.append(ad)
    
    # fig,ax = plt.subplots()
    # bp = ax.boxplot(yup)
    # plt.xlabel('penguin species')
    # plt.ylabel('bill length (mm)')
    # plt.xticks([1, 2, 3], ['Adelie', 'Chinstrap', 'Gentoo'])
    # plt.title("Penguin Bill Length Based on Species")
    # plt.show()
    """scatter plot
    """
    #color is penguin species
    #one axis is one of the other values
    #1=ade, 2=chin, 3=gentoo
    # df = remove_na(df, "bill_length_mm")
    # df = remove_na(df, "bill_depth_mm")
    # df = remove_na(df, "flipper_length_mm")
    # df = remove_na(df, "body_mass_g")
    # x1 = df[df['species']=='Adelie']['body_mass_g']
    # x2 = df[df['species']=='Chinstrap']['body_mass_g']
    # x3 = df[df['species']=='Gentoo']['body_mass_g']

    # y1 = df[df['species']=='Adelie']['flipper_length_mm']
    # y2 = df[df['species']=='Chinstrap']['flipper_length_mm']
    # y3 = df[df['species']=='Gentoo']['flipper_length_mm']
    # print(df)
    # plt.scatter(x1,y1,c = "red", label= "Adelie")
    # plt.scatter(x2,y2,c = "blue",label= "Chinstrap")
    # plt.scatter(x3,y3,c = "green",label= "Gentoo")

    # plt.xlabel('body mass (g)')
    # plt.ylabel('flipper length (mm)')
    # plt.title("Penguin Body Mass vs Flipper Length")
    # plt.legend()
    # plt.show()

    # ya = remove_na(ya, "sex")
    # ya = remove_na(ya, "year")
    #plt.hist(df_wNA[['Adelie','Chinstrap','Gentoo']],bins=1)  
    #lt.hist(df_noNA[['Adelie','Chinstrap','Gentoo']],bins=1)
    #plt.hist((ad,ch,gen), histtype='bar',label="Removed NaN values")
    #plt.hist((ad1,ch1,gen1), histtype='bar',label="with NaN values")
    

    # Load data
    df = load_csv("data/penguins.csv")
    
    #Remove NA
    df = remove_na(df, "species")

    # One hot encoding
    df = onehot(df, "species")

    # Convert to numeric
    df_np = to_numeric(df)

if __name__ == "__main__":
    main()
