from rich import print
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from app.utils.types import Dataset

from utils.types import RecommendResult, DFData, DFMovieRatingPred, DFUserMoviesPred
from recommenders.base_recommend import BaseRecommender


OBJECTIVE_VAR = 'rating'
STATS_FEATURES = [
    'rating_mean_by_user_id',
    'rating_min_by_user_id',
    'rating_max_by_user_id',
    'rating_mean_by_movie_id',
    'rating_min_by_movie_id',
    'rating_max_by_movie_id',
]
GENRE_FEATURES = [
    'is_IMAX',
    'is_Mystery',
    'is_Animation',
    'is_Musical',
    'is_Thriller',
    'is_Sci-Fi',
    'is_Western',
    'is_Romance',
    'is_Adventure',
    'is_Film-Noir',
    'is_Crime',
    'is_War',
    'is_Action',
    'is_Horror',
    'is_Fantasy',
    'is_Comedy',
    'is_Documentary',
    'is_Children',
    'is_(no genres listed)',
    'is_Drama',
]
FEATURES = STATS_FEATURES + GENRE_FEATURES


class RandomForestRecommender(BaseRecommender):
    """
    A class used to recommend movies based on a Random Forest model.

    The model is trained on a dataset with features including statistical features and movie genre features.

    Attributes
    ----------
    dataset : Dataset
        The dataset containing the movie ratings and other features.

    Methods
    -------
    recommend() -> RecommendResult
        Trains the model, makes predictions, and returns the recommended movies for each user.
    """
    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset)

    def recommend(self) -> RecommendResult:
        """
        Trains the model, makes predictions, and returns the recommended movies for each user.

        Returns
        -------
        RecommendResult
            The predicted movie ratings and the top 10 recommended movies for each user.
        """
        df_train, df_test, df_train_all = self._calc_features()

        # モデルの学習
        X_train: np.ndarray = df_train[FEATURES].to_numpy()
        y_train: np.ndarray = df_train[OBJECTIVE_VAR].to_numpy()
        model: RandomForestRegressor = RandomForestRegressor(n_jobs=-1, random_state=0)
        model.fit(X=X_train, y=y_train)

        # testデータ予測
        X_test: np.ndarray = df_test[FEATURES].to_numpy()
        test_pred: np.ndarray = model.predict(X=X_test)
        df_movie_rating_pred: DFMovieRatingPred = (
            df_test
            .with_columns(rating_pred=pl.lit(test_pred))
            .select(['user_id', 'movie_id', 'rating', 'rating_pred'])
        )

        # レコメンド用データ予測
        X_train_all: np.ndarray = df_train_all[FEATURES].to_numpy()
        train_all_pred: np.ndarray = model.predict(X=X_train_all)
        df_user_evaluated_movies: pl.DataFrame = (
            self.dataset.df_train
            .group_by('user_id')
            .agg(movie_id_evaluated=pl.col('movie_id'))
        )
        df_train_all = (
            df_train_all
            .select(['user_id', 'movie_id'])
            .with_columns(rating_pred=pl.lit(train_all_pred))
            .sort(by=['user_id', 'rating_pred'], descending=[False, True])  # ユーザーごとにratingが高いmovie順にソート
            .group_by('user_id')
            .agg(movie_id_pred=pl.col('movie_id'))  # rating_predが高い順で並ぶmovie_idのリストを作成
            .join(df_user_evaluated_movies, on='user_id', how='left')
            .with_columns(  # ユーザーが学習データ内で評価していない映画の中からrating_predが高い順に10件
                pl.col('movie_id_pred')
                .list.set_difference(pl.col('movie_id_evaluated'))
                .list.head(10)
                .alias('movie_id_pred')
            )
        )
        df_user_movies_pred: DFUserMoviesPred = (
            self.dataset.df_user_movies_test
            .join(df_train_all, on='user_id', how='left')
            .select(['user_id', 'movie_id', 'movie_id_pred'])
        )
        return RecommendResult(df_movie_rating_pred=df_movie_rating_pred, df_user_movies_pred=df_user_movies_pred)

    def _calc_features(self) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Calculates the features for the model.

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
            The training data, test data, and all data with the calculated features.
        """
        df_train: DFData = self.dataset.df_train
        df_test: DFData = self.dataset.df_test

        df_user: pl.Series = df_train.select('user_id').unique()
        df_movies: pl.Series = df_train.select('movie_id').unique()
        df_train_all: pl.DataFrame = df_user.join(df_movies, how='cross')  # レコメンド用に全ての user x movie のパターンを作成

        # 統計量特徴量を作成
        df_user_stats: pl.DataFrame = self._calc_statistics(grouping_col='user_id')
        df_movie_stats: pl.DataFrame = self._calc_statistics(grouping_col='movie_id')
        df_train: pl.DataFrame = (
            df_train
            .select(['user_id', 'movie_id', 'rating'])
            .join(df_user_stats, on='user_id', how='left')
            .join(df_movie_stats, on='movie_id', how='left')
        )
        df_test: pl.DataFrame = (
            df_test
            .select(['user_id', 'movie_id', 'rating'])
            .join(df_user_stats, on='user_id', how='left')
            .join(df_movie_stats, on='movie_id', how='left')
            .fill_null(df_train['rating'].mean())  # testデータのみに存在する組み合わせはtrainデータの平均値で代替する
            .with_columns(user_id=pl.col('user_id').cast(int))  # fill_null を行うと floatとなるため、型をintに戻す
            .with_columns(movie_id=pl.col('movie_id').cast(int))
        )
        df_train_all: pl.DataFrame = (
            df_train_all
            .select(['user_id', 'movie_id'])
            .join(df_user_stats, on='user_id', how='left')
            .join(df_movie_stats, on='movie_id', how='left')
        )

        # 映画ジャンル特徴量を作成
        df_mc: pl.DataFrame = self.dataset.df_movie_content.select(['movie_id', 'genre'])
        unique_genres: pl.Series = df_mc['genre'].list.explode().unique()
        for genre in unique_genres:
            df_mc = (
                df_mc
                .with_columns(
                    pl.col('genre')
                    .list.contains(genre)
                    .alias(f'is_{genre}')
                )
            )
        df_mc = df_mc.drop('genre')
        df_train = df_train.join(df_mc, on='movie_id', how='left')
        df_test = df_test.join(df_mc, on='movie_id', how='left')
        df_train_all = df_train_all.join(df_mc, on='movie_id', how='left')

        return df_train, df_test, df_train_all

    def _calc_statistics(self, grouping_col: str) -> pl.DataFrame:
        """
        Calculates statistical features for the model.

        Parameters
        ----------
        grouping_col : str
            The column to group by when calculating the statistics.

        Returns
        -------
        pl.DataFrame
            The DataFrame with the calculated statistical features.
        """
        df: pl.DataFrame = (
            self.dataset.df_train
            .group_by(grouping_col)
            .agg(
                pl.col('rating').mean().name.suffix(f'_mean_by_{grouping_col}'), 
                pl.col('rating').min().name.suffix(f'_min_by_{grouping_col}'), 
                pl.col('rating').max().name.suffix(f'_max_by_{grouping_col}'),   
            )
        )
        return df
