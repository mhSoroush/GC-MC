import argparse
import numpy as np
import pandas as pd
import random
import scipy.sparse as sp
import pickle

SEP = dict({'book': ';'})
files = ['/BX-Users.csv', '/converted_book.csv', '/converted_rating.csv']
dataset = 'book'
data_dir = './data/' + dataset
num_records = 80000


def map_data(data):
    """
    Map data to proper indices in case they are not in a continues [0, N) range

    Parameters
    ----------
    data : np.int64 arrays

    Returns
    -------
    mapped_data : np.int64 arrays
    n : length of mapped_data

    """
    uniq = list(set(data))

    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array(map(lambda x: id_dict[x], data))
    n = len(uniq)

    return data, id_dict, n


def user_feature_extraction(num_users, u_dict):
    print("User feature extraction...")
    filename = './data/' + dataset + files[0]
    name_headers = ['user_id', 'location', 'age']
    dtypes = {
        'user_id': np.int64, 'location': str,
        'age': np.int32}
    user_df = pd.read_csv(filename, sep=SEP[dataset], header=None, names=name_headers,
                          error_bad_lines=False, skiprows=1, warn_bad_lines=False)

    cols = ['location']
    cntr = 0
    feat_dicts = []
    for header in cols:
        d = dict()
        if header == 'location':
            feats = user_df['location'].values.astype(dtypes['location']).tolist()
            feats = [loc.split(',')[-1].lower().strip() for loc in feats]
            feats = list(set(feats))
        d.update({f: i for i, f in enumerate(feats, start=cntr) if f != ''})
        feat_dicts.append(d)
        cntr += len(d)

    num_feats = sum(len(d) for d in feat_dicts)

    # Creating 0 or 1 valued features for all features
    u_features = np.zeros((num_users, num_feats), dtype=np.float32)
    for _, row in user_df.iterrows():
        u_id = row['user_id']

        if u_id in u_dict.keys():
            for k, header in enumerate(cols):
                location = str(row[header]).split(',')[-1].lower().strip()
                if location != '':
                    u_features[u_dict[u_id], feat_dicts[k][location]] = 1.

    return u_features


def book_feature_extraction(num_items, v_dict):
    print("Item feature extraction...")
    # Item/book feature extraction
    filename = data_dir + files[1]
    name_headers = ['book_id', 'book_title', 'book_author', 'publication_year', 'publisher']
    dtypes = {
        'book_id': np.int64, 'book_title': str,
        'book_author': str, 'publication_year': np.int32, 'publisher': str}
    book_df = pd.read_csv(filename, sep=SEP[dataset], header=None, names=name_headers,
                          error_bad_lines=False, skiprows=1, warn_bad_lines=False)

    # Insert columns for feature extraction
    v_cols = ['book_author']

    cntr = 0
    feat_dicts = []
    for header in v_cols:
        d = dict()
        if header == 'book_author':
            feats = book_df[header].values.astype(dtypes['book_author']).tolist()
            feats = [a.lower().strip() for a in feats]
            feats = list(set(feats))
        d.update({f: i for i, f in enumerate(feats, start=cntr)})
        feat_dicts.append(d)
        cntr += len(d)

    num_feats = sum(len(d) for d in feat_dicts)

    # Creating 0 or 1 valued features for all features
    v_features = np.zeros((num_items, num_feats), dtype=np.float32)
    for _, row in book_df.iterrows():
        v_id = row['book_id']

        if v_id in v_dict.keys():
            for k, header in enumerate(v_cols):
                book_author = str(row[header]).strip().lower()
                v_features[v_dict[v_id], feat_dicts[k][book_author]] = 1.

    return v_features


def load_data(seed=1234, verbose=True):
    # type: (list, np.int32, bool) -> object
    if dataset == 'book':
        # Load rating
        filename = data_dir + files[2]
        name_header = ['u_nodes', 'v_nodes', 'ratings']
        dtypes = {'u_nodes': np.int64, 'v_nodes': np.int64, 'ratings': np.float32}
        data = pd.read_csv(filename, sep=SEP[dataset], header=None, names=name_header,
                           error_bad_lines=False, skiprows=1, warn_bad_lines=False)

        # shuffle here like cf-nade paper with python's own random class
        # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
        data_array = data.as_matrix().tolist()
        # We have 1031175 records in rating file, but creating such a matrix is out of memory.
        # Therefore, we take 100000 records

        #data_array = data_array[: num_records]

        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int64)
        ratings = ratings.astype(np.float32)

        v_features = book_feature_extraction(num_items, v_dict)
        v_features = sp.csc_matrix(v_features)

        # Load user features
        u_features = user_feature_extraction(num_users, u_dict)
        u_features = sp.csc_matrix(u_features)

        print ("Feature extraction processes are finished")
        return num_users, num_items, u_nodes_ratings, v_nodes_ratings, ratings, u_features, v_features

