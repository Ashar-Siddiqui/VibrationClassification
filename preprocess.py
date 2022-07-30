from os import extsep, mkdir
from os.path import dirname, join, exists
import sys

import numpy as np
import scipy.stats as st

def main():
    current_directory = dirname(__name__)
    kaggle_data_filenames = [fi for i in range(5) for letter in ('D', 'E') if exists(join(current_directory, (fi:=f'{i}{letter}')) + f'{extsep}csv')]
    print(f'Data files found from Kaggle: {kaggle_data_filenames}')

    directory_for_processed_data = join(current_directory, f'data')
    if len(kaggle_data_filenames) != 0 and not exists(directory_for_processed_data):
        print(f'Creating directory "{directory_for_processed_data}" for processed data...')
        mkdir(directory_for_processed_data)

    for filename in kaggle_data_filenames:
        input_filename = join(current_directory, f'{filename}{extsep}csv')
        output_filename = join(directory_for_processed_data, f'{filename}{extsep}npy')
        if not exists(output_filename):
            np.save(output_filename, process(input_filename), allow_pickle=False)
        else:
            print(
                f'The file "{output_filename}" already exists! '
                f'Please delete this file in order to override.',
                file=sys.stderr,
            )

'''
This function reads in a csv `filename` as a 2D matrix.
The header (which consists of column names) are ignored
as well as the first `skiprows` rows. These are rows
that correspond to the motor starting up, and therefore
they are not useful data to us.

We batch together each group of `batchsize` rows,
and then compute statistics across several
columns in each batch. Each of these statistics is 
concatenated into an array. This array is like a
"representative summary" for each `batchsize`
rows.
'''
def process(filename, batchsize=4096, skiprows=50000):
    print(f'Loading {filename}...')
    d = np.genfromtxt(filename, delimiter=',', skip_header=1 + skiprows, dtype=np.float32)

    # This forces the number of rows to be a multiple of batchsize, and ignores the remainder
    numrows = d.shape[0]
    truncated_numrows = numrows // batchsize * batchsize
    d = d[:truncated_numrows,:]

    # Reshape to have batches. This is so these batches can be averaged across
    d = d.reshape(truncated_numrows // batchsize, batchsize, -1)
    rpm = d[:,1,:]
    vibration1 = d[:,2,:]
    vibration2 = d[:,3,:]
    vibration3 = d[:,4,:]
    print(f'Processing {d.shape[0] * d.shape[1]} rows from {filename}...')
    return np.stack([
        np.mean(rpm, axis=-1),
        np.std(vibration1, axis=-1),
        st.kurtosis(vibration1, axis=-1),
        np.std(vibration2, axis=-1),
        st.kurtosis(vibration2, axis=-1),
        np.std(vibration3, axis=-1),
        st.kurtosis(vibration3, axis=-1),
    ], axis=1)

if __name__ == '__main__':
    main()
