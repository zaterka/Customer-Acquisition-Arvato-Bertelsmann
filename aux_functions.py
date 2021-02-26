import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle


from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import numpy as np



def get_unknown(attributes):
    na = attributes.ffill()
    na = na.loc[(na['Meaning'].str.contains('unknown')) | (na['Meaning'] == 'no transaction known') | (na['Meaning'] == 'no transactions known')]
    na = pd.concat([pd.Series(row['Attribute'], str(row['Value']).split(',')) for _, row in na.iterrows()]).reset_index()
    na.rename(columns={'index':'Value', 0: 'Attribute'}, inplace = True)
    return na

def pre_process_df(df: pd.DataFrame, columns: list, attributes: pd.DataFrame, filter_na = True) -> pd.DataFrame:

    attribute_na = get_unknown(attributes)

    if filter_na:
        columns_keep = []
        for column in columns:
            na_values = attribute_na.loc[attribute_na['Attribute'] == column, 'Value'].values
            df.loc[df[column].isin(na_values), column] = np.NaN
            if (df[column].isna().sum()/len(df[column])) <= 0.29:
                columns_keep.append(column)

        df = df[columns_keep]
        df.drop(columns = ['D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM', 'CAMEO_DEU_2015'], inplace = True)
    else:
        
        df = df[columns]

    df['cut_row'] = df.isna().sum(axis=1)/df.shape[1]
    df = df.loc[df['cut_row'] <= 0.7]
    df.drop(columns = ['cut_row', 'LNR'], inplace=True)
    

    for column in ['CAMEO_DEUG_2015', 'CAMEO_INTL_2015']:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    ost_west_dict = {'W': 0, 'O': 1}
    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].map(ost_west_dict)
    
    
    return df 


def fill_bool(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if df[column].nunique() == 2:
            df[column] = df[column].fillna(ref_df[column].mode()[0]) 
    
    return df


def fill_num(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        if df[column].nunique() != 2:
            df[column] = df[column].fillna(ref_df[column].dropna().median())
            
    return df





def compute_bic(kmeans,X):
    '''
    Computes the BIC metric for a given clusters

    Arguments:
    kmeans:  List of clustering object from scikit learn
    X:  multidimension np array of data points

    Returns:
    BIC value
   '''
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return BIC





def pre_process_supervised(df, attributes, reference ,  train=True, kmeans_path = 'kmeans_7.pkl'):
    attribute_na = get_unknown(attributes)

    chosen_k = pickle.load(open(kmeans_path, 'rb'))
    columns_keep = []
    for column in df.columns:
        na_values = attribute_na.loc[attribute_na['Attribute'] == column, 'Value'].values
        df.loc[df[column].isin(na_values), column] = np.NaN
        if (df[column].isna().sum()/len(df[column])) <= 0.29:
            columns_keep.append(column)

    df = df[columns_keep]
    df.drop(columns = ['EINGEFUEGT_AM',  'LNR'], inplace = True)

    if train:
        df['cut_row'] = df.isna().sum(axis=1)/df.shape[1]
        df = df.loc[df['cut_row'] <= 0.73]
        df.drop(columns = ['cut_row'], inplace=True)
    

    for column in ['CAMEO_DEUG_2015', 'CAMEO_INTL_2015']:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    ost_west_dict = {'W': 0, 'O': 1}
    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].map(ost_west_dict)

    df = pd.get_dummies(df, columns=['CAMEO_DEU_2015', 'D19_LETZTER_KAUF_BRANCHE'], dummy_na=True)
    
    df = fill_bool(df,df)
    df = fill_num(df, df)

    # Add cluster for each line

    scaler = MinMaxScaler()
    df_scaled = df[list(reference.columns)]
    df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=df_scaled.columns)
    pca = PCA(n_components=200)
    df_scaled = pca.fit_transform(df_scaled)
    clusters = chosen_k.predict(df_scaled)
    del df_scaled, scaler, pca
    df['CLUSTER'] = clusters
    df = pd.get_dummies(df, columns=['CLUSTER'])
    
    if train:
        # put RESPONSE column last
        ordered_columns = list(df.columns)
        ordered_columns.remove('RESPONSE')
        ordered_columns = ['RESPONSE'] + ordered_columns
        df = df[ordered_columns]


    return df 


if __name__ == '__main__':

    azdias = pd.read_csv('./data/azdias.csv', sep=';', index_col=0)
    attributes = pd.read_excel(r'./data/DIAS Attributes - Values 2017.xlsx', skiprows=1).iloc[:,1:]


    azdias = pre_process_df(azdias, list(azdias.columns), attributes)

    azdias = fill_bool(azdias, azdias)

    azdias = fill_num(azdias, azdias)

    scaler = MinMaxScaler()
    azdias_scaled = pd.DataFrame(scaler.fit_transform(azdias), columns = azdias.columns)

    pca = PCA(n_components=200)
    pca.fit(azdias_scaled)

    with open('pca.pkl', 'wb') as pickle_file:
        pickle.dump(pca, pickle_file)
