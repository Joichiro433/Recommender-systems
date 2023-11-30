from typing import TypeAlias
from pathlib import Path

import polars as pl
from nptyping import DataFrame, Structure as S

from utils.types import Dataset, DFMovieTag, DFRatingMovie, DFData


DFMovie: TypeAlias = pl.DataFrame | DataFrame[S[
    """
    movie_id: Int, 
    title: Str, 
    genre: List[Str]
    """
]]

DFTag: TypeAlias = pl.DataFrame | DataFrame[S[
    """
    movie_id: Int, 
    tag: List[Str]
    """
]]

DFRating: TypeAlias = pl.DataFrame | DataFrame[S[
    """
    user_id: Int, 
    movie_id: Int, 
    rating: Float, 
    timestamp: Int
    """
]]


class DataLoader:
    """
    A class used to load and preprocess movie-related data.

    Attributes
    ----------
    num_users : int
        The number of users to consider in the dataset.
    num_test_items : int
        The number of test items to consider in the dataset.
    data_path : Path
        The path to the directory containing the data.

    Methods
    -------
    load() -> Dataset
        Loads the data, preprocesses it, splits it into training and testing sets, and returns a `Dataset` object.
    """
    def __init__(
            self, 
            num_users: int = 1000,
            num_test_items: int = 5,
            data_path: Path = Path('./data/')
        ) -> None:
        self.num_users: int = num_users
        self.num_test_items: int = num_test_items
        self.data_path: Path = data_path

    def load(self) -> Dataset:
        """
        Loads the data, preprocesses it, splits it into training and testing sets, and returns a `Dataset` object.

        Returns
        -------
        Dataset
            The loaded and preprocessed dataset.
        """
        df_rating_with_movie, df_movie_with_tag = self._preprocess()
        df_train, df_test = self._split_data(df_rating_with_movie=df_rating_with_movie)

        # ranking用の評価データは、各ユーザーの評価値が4以上の映画だけを正解とする
        df_user_movies_test: pl.DataFrames = (
            df_test
            .filter(pl.col('rating') >= 4)
            .group_by('user_id')
            .agg('movie_id')
        )

        return Dataset(
            df_train=df_train, 
            df_test=df_test, 
            df_user_movies_test=df_user_movies_test,
            df_movie_content=df_movie_with_tag)

    def _split_data(self, df_rating_with_movie: DFRatingMovie) -> tuple[DFData, DFData]:
        """
        Splits the data into training and testing sets based on the user's most recent ratings.

        Parameters
        ----------
        df_rating_with_movie : DFRatingMovie
            The DataFrame containing the movie ratings.

        Returns
        -------
        tuple[DFData, DFData]
            The training and testing data.
        """
        # 学習用とテスト用にデータを分割する
        # 各ユーザの直近の５件の映画を評価用に使い、それ以外を学習用とする
        df: DFData = (
            df_rating_with_movie
            .sort('movie_id')
            .with_columns(
                pl.col('timestamp')
                .rank(method='ordinal', descending=True)  # 直近評価した映画から順番を付与
                .over('user_id')
                .alias('rating_order')
            )
        )
        # 各ユーザの直近の５件の映画を評価用に使い、それ以外を学習用とする
        df_train: DFData = df.filter(pl.col('rating_order') > self.num_test_items) 
        df_test: DFData = df.filter(pl.col('rating_order') <= self.num_test_items)
        return df_train, df_test

    def _preprocess(self) -> tuple[DFRatingMovie, DFMovieTag]:
        """
        Reads the data from Parquet files, preprocesses it (e.g., splitting genres, converting tags to lowercase), and joins the dataframes together.

        Returns
        -------
        tuple[DFRatingMovie, DFMovieTag]
            The preprocessed movie ratings and movie tags.
        """
        df_movie: DFMovie = (
            pl.read_parquet(self.data_path / 'movies.parquet')
            .with_columns(genre=pl.col('genre').str.split('|'))
        )
        df_tag: DFTag = (
            pl.read_parquet(self.data_path / 'tags.parquet')
            .with_columns(tag=pl.col('tag').str.to_lowercase())
            .group_by('movie_id')
            .agg(pl.col('tag'))
        )
        df_rating: DFRating = pl.read_parquet(self.data_path / 'ratings.parquet')
        valid_user_ids: list[int] = sorted(df_rating['user_id'].unique())[:self.num_users]
        df_rating = df_rating.filter(pl.col('user_id') <= max(valid_user_ids))

        df_movie_with_tag: DFMovieTag = df_movie.join(df_tag, on='movie_id', how='left')
        df_rating_with_movie: DFRatingMovie = df_rating.join(df_movie_with_tag, on='movie_id', how='inner')

        return df_rating_with_movie, df_movie_with_tag
