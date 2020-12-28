from MetalnessCalculator import MetalnessCalculator


import pandas as pd
from MetalnessCalculator import MetalnessCalculator

dataset_eng_df = pd.read_csv('../mpi_py_example/data/dark_lyrics.csv.4', sep=',', escapechar='\\')
no_metal_dataset_eng_df = pd.read_csv('../mpi_py_example/data/light_lyrics.csv.1', encoding='utf-8', sep=',', dtype=str,escapechar='\\')

calc = MetalnessCalculator(dataset_eng_df,no_metal_dataset_eng_df)
