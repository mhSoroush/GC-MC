import argparse
import numpy as np
import pandas as pd
import csv
import os

# map book item to id
dataset = 'book'
SEP = dict({'book': ';'})
files = ['/BX-Book-Ratings.csv', '/BX-Users.csv', '/BX-Books.csv']
cleaned_files = ['/converted_rating.csv', '/converted_book.csv']
data_dir = './data/' + dataset

def data_iterator(data, batch_size):
    """
    A simple data iterator from https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
    :param data: list of numpy tensors that need to be randomly batched across their first dimension.
    :param batch_size: int, batch_size of data_iterator.
    Assumes same first dimension size of all numpy tensors.
    :return: iterator over batches of numpy tensors
    """
    # shuffle labels and features
    max_idx = len(data[0])
    idxs = np.arange(0, max_idx)
    np.random.shuffle(idxs)
    shuf_data = [dat[idxs] for dat in data]

    # Does not yield last remainder of size less than batch_size
    for i in range(max_idx//batch_size):
        data_batch = [dat[i*batch_size:(i+1)*batch_size] for dat in shuf_data]
        yield data_batch


def is_cleaned_data_saved():
    if os.path.isfile(data_dir + cleaned_files[0]) and os.path.isfile(data_dir + cleaned_files[1]):
        return True
    return False


def read_ratings_file(dataset):
    """
    Read BX-Book-Ratings.csv file
    """
    filename = data_dir + files[0]
    name_headers = ['user_id', 'isbn', 'rating']
    rating_df = pd.read_csv(filename, sep=SEP[dataset], header=None, names=name_headers,
                            error_bad_lines=False, skiprows=1, warn_bad_lines=False)
    return rating_df


def read_book_file(dataset):
    name_headers = ['isbn', 'book_title', 'book_author', 'publication_year',
                    'publisher', 'url_s', 'url_m', 'url_large']
    filename = data_dir + files[2]
    book_df = pd.read_csv(filename, sep=SEP[dataset], header=None, names=name_headers,
                          error_bad_lines=False, skiprows=1, warn_bad_lines=False)
    return book_df


def read_user_file(dataset):
    filename = data_dir + files[1]
    name_headers = ['user_id', 'location', 'age']
    dtypes = {
        'user_id': np.int64, 'location': str, 'age': np.int32}
    user_df = pd.read_csv(filename, sep=SEP[dataset], header=None, names=name_headers,
                          error_bad_lines=False, skiprows=1, warn_bad_lines=False)
    return user_df


def cleaning_rating(dataset_name, rating_df, user_ids, isbn_ids, isbn_dicts):
    print("Rating cleaning process...")
    cleaned_rating = []
    for _, row in rating_df.iterrows():
        u_id = row['user_id']
        v_id = row['isbn']
        r = float(row['rating'])
        if u_id in user_ids and v_id in isbn_ids and r > 0:
            row = list(row)
            row[1] = isbn_dicts[v_id]
            cleaned_rating.append(row)

    write_cleaned_rating_2_new_file(dataset_name, cleaned_rating)


def write_cleaned_rating_2_new_file(dataset, book_rating_list):
    print ("Writing cleaned rating file...")
    write_file = data_dir + cleaned_files[0]

    file_writer = open(write_file, 'w')
    # write only the headers
    writer = csv.writer(file_writer, delimiter=';')
    writer.writerow(['user_id', 'book_id', 'rating'])

    writer.writerows(book_rating_list)
    print('finished book-rating writing')


def convert_isbn2id_in_book_file(dataset, book_df, isbn_dicts):
    print ("Convert isbn to id and save the converted file")
    write_file = data_dir + cleaned_files[1]

    file_writer = open(write_file, 'w')
    # write only the headers
    writer = csv.writer(file_writer, delimiter=';')
    writer.writerow(['book_id', 'book_title', 'book_author', 'publication_year', 'publisher'])

    for _, row in book_df.iterrows():
        if row['isbn'] in isbn_dicts.keys():
            book_id = isbn_dicts[row['isbn']]
            book_title = row['book_title']
            book_author = row['book_author']
            publication_year = row['publication_year']
            publisher = row['publisher']
            writer.writerow([book_id, book_title, book_author, publication_year, publisher])

    print('finished writing')


def cleaning_data(dataset):
    # Reading files
    rating_df = read_ratings_file(dataset)
    book_df = read_book_file(dataset)
    user_df = read_user_file(dataset)

    # Get all ISBNs
    isbn_ids = book_df['isbn'].values.tolist()
    isbn_ids = map(lambda x: str(x).strip(), isbn_ids)
    isbn_dicts = {key: ids for ids, key in enumerate(isbn_ids)}

    # Get all user ids
    user_ids = user_df['user_id'].values.tolist()

    # Keep rows which exits in User_ids and isbn_ids
    cleaning_rating(dataset, rating_df, user_ids, isbn_ids, isbn_dicts)

    # convert isbn to id in book file
    convert_isbn2id_in_book_file(dataset, book_df, isbn_dicts)

    print ("Cleaning process done!")


if __name__ == '__main__':
    np.random.seed(222)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='book', help='which dataset to preprocess')
    args = parser.parse_args()

    cleaning_data(args.dataset)

    # logging.info("data %s preprocess: done.",args.dataset)
