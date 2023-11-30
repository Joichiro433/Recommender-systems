from pathlib import Path
import subprocess

import pandas as pd
import polars as pl


def download_dataset():
    # wget command
    wget_command = "wget -nc --no-check-certificate https://files.grouplens.org/datasets/movielens/ml-10m.zip -P ./data/"
    subprocess.run(wget_command, shell=True, check=True)

    # unzip command
    unzip_command = "unzip -n ./data/ml-10m.zip -d ./data/"
    subprocess.run(unzip_command, shell=True, check=True)


def convert_to_parquet():
    # 映画の情報の読み込み(10681作品)
    m_cols = ['movie_id', 'title', 'genre']  # movieIDとタイトル名のみ使用
    movies = pd.read_csv('./data/ml-10M100K/movies.dat', names=m_cols, sep='::' , encoding='latin-1', engine='python')
    movies = pl.from_pandas(movies)

    # ユーザが付与した映画のタグ情報の読み込み
    t_cols = ['user_id', 'movie_id', 'tag', 'timestamp']
    user_tagged_movies = pd.read_csv('../data/ml-10M100K/tags.dat', names=t_cols, sep='::', engine='python')
    user_tagged_movies = pl.from_pandas(user_tagged_movies)

    # 評価値データの読み込み
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('../data/ml-10M100K/ratings.dat', names=r_cols, sep='::', engine='python')
    ratings = pl.from_pandas(ratings)

    movies.write_parquet('./data/movies.parquet')
    user_tagged_movies.write_parquet('./data/tags.parquet')
    ratings.write_parquet('./data/ratings.parquet')


if __name__ == '__main__':
    if not Path('./data/ml-10M100K/').exists():
        download_dataset()
    convert_to_parquet()
