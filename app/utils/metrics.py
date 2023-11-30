import numpy as np
import polars as pl
from sklearn.metrics import mean_squared_error

from utils.types import Metrics, DFMovieRatingPred, DFUserMoviesPred


class MetricCalculator:
    """
    A class used to calculate various metrics for evaluating the performance of a recommendation system.

    Class Methods
    -------------
    calc(cls, df_movie_rating_pred: DFMovieRatingPred, df_user_movies_pred: DFUserMoviesPred, k: int) -> Metrics
        Calculates the root mean square error (RMSE), precision at k, and recall at k, and returns them as a `Metrics` object.
    """
    @classmethod
    def calc(
            cls, 
            df_movie_rating_pred: DFMovieRatingPred,
            df_user_movies_pred: DFUserMoviesPred,
            k: int,
        ) -> Metrics:
        """
        Calculates the root mean square error (RMSE), precision at k, and recall at k, and returns them as a `Metrics` object.

        Parameters
        ----------
        df_movie_rating_pred : DFMovieRatingPred
            DataFrame containing the true and predicted movie ratings.
        df_user_movies_pred : DFUserMoviesPred
            DataFrame containing the user movies predictions.
        k : int
            The number of top recommendations to consider.

        Returns
        -------
        Metrics
            A Metrics object containing the RMSE, precision at k, and recall at k.
        """
        rmse: float = cls._calc_rmse(df_movie_rating_pred=df_movie_rating_pred)
        precision_at_k: float = cls._calc_precision_at_k(df_user_movies=df_user_movies_pred, k=k)
        recall_at_k: float = cls._calc_recall_at_k(df_user_movies=df_user_movies_pred, k=k)
        return Metrics(rmse=rmse, precision_at_k=precision_at_k, recall_at_k=recall_at_k)
        
    @classmethod
    def _calc_rmse(
            cls, 
            df_movie_rating_pred: DFMovieRatingPred,
        ) -> float:
        """
        Calculates the root mean square error (RMSE) between the true and predicted movie ratings.

        Parameters
        ----------
        df_movie_rating_pred : DFMovieRatingPred
            DataFrame containing the true and predicted movie ratings.

        Returns
        -------
        float
            The calculated RMSE.
        """
        true_rating = df_movie_rating_pred['rating']
        pred_rating = df_movie_rating_pred['rating_pred']
        return np.sqrt(mean_squared_error(y_true=true_rating, y_pred=pred_rating))

    @classmethod
    def _calc_precision_at_k(
            cls, 
            df_user_movies: DFUserMoviesPred,
            k: int,
        ) -> float:
        """
        Calculates the precision at k, which is the proportion of recommended items in the top-k that are relevant.

        Parameters
        ----------
        df_user_movies : DFUserMoviesPred
            DataFrame containing the user movies predictions.
        k : int
            The number of top recommendations to consider.

        Returns
        -------
        float
            The calculated precision at k.
        """
        df_user_precision = (
            df_user_movies
            .with_columns(movie_id_pred=pl.col('movie_id_pred').list.head(k))
            .with_columns(corret_movie_id=pl.col('movie_id').list.set_intersection(pl.col('movie_id_pred')))
            .with_columns(precision=pl.col('corret_movie_id').list.len() / k)
        )
        return df_user_precision['precision'].mean()
    
    @classmethod
    def _calc_recall_at_k(
            cls, 
            df_user_movies: DFUserMoviesPred,
            k: int,
        ) -> float:
        """
        Calculates the recall at k, which is the proportion of relevant items that are found in the top-k recommendations.

        Parameters
        ----------
        df_user_movies : DFUserMoviesPred
            DataFrame containing the user movies predictions.
        k : int
            The number of top recommendations to consider.

        Returns
        -------
        float
            The calculated recall at k.
        """
        df_user_recall = (
            df_user_movies
            .with_columns(movie_id_pred=pl.col('movie_id_pred').list.head(k))
            .with_columns(corret_movie_id=pl.col('movie_id').list.set_intersection(pl.col('movie_id_pred')))
            .with_columns(recall=pl.col('corret_movie_id').list.len() / pl.col('movie_id').list.len())
        )
        return df_user_recall['recall'].mean()
    
    @classmethod
    def _calc_fscore_at_k(
            cls, 
            df_user_movies: DFUserMoviesPred,
            k: int,
        ) -> float:
        """
        Calculates the F-score at k, which is the harmonic mean of precision at k and recall at k.

        Parameters
        ----------
        df_user_movies : DFUserMoviesPred
            DataFrame containing the user movies predictions.
        k : int
            The number of top recommendations to consider.

        Returns
        -------
        float
            The calculated F-score at k.
        """
        precision_at_k: float = cls._calc_precision_at_k(df_user_movies=df_user_movies, k=k)
        recall_at_k: float = cls._calc_recall_at_k(df_user_movies=df_user_movies, k=k)
        fscore_at_k: float = (2 * recall_at_k * precision_at_k) / (recall_at_k + precision_at_k)
        return fscore_at_k
