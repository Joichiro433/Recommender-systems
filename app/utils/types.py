from typing import TypeAlias
from dataclasses import dataclass

from nptyping import DataFrame, Structure as S
import polars as pl


DFMovieTag: TypeAlias = pl.DataFrame | DataFrame[S[
    """
    movie_id: Int, 
    title: Str, 
    genre: List[Str],
    tag: List[Str]
    """
]]

DFRatingMovie: TypeAlias = pl.DataFrame | DataFrame[S[
    """
    user_id: Int,
    movie_id: Int,
    rating: Float,
    timestamp: Int,
    title: Str, 
    genre: List[Str],
    tag: List[Str]
    """
]]

DFData: TypeAlias = pl.DataFrame | DataFrame[S[
    """
    user_id: Int,
    movie_id: Int,
    rating: Float,
    timestamp: Int,
    title: Str, 
    genre: List[Str],
    tag: List[Str],
    rating_order: Int
    """
]]

DFMovieRatingPred: TypeAlias = pl.DataFrame | DataFrame[S[
    """
    user_id: Int,
    movie_id: Int,
    rating: Float,
    rating_pred: Float
    """
]]

DFUserMovies: TypeAlias = pl.DataFrame | DataFrame[S[
    """
    user_id: Int,
    movie_id: List[Int]
    """
]]

DFUserMoviesPred: TypeAlias = pl.DataFrame | DataFrame[S[
    """
    user_id: Int,
    movie_id: List[Int],
    movie_id_pred: List[Int]
    """
]]


@dataclass(frozen=True)
class Dataset:
    df_train: DFData
    df_test: DFData
    df_user_movies_test: DFUserMovies
    df_movie_content: DFMovieTag


@dataclass(frozen=True)
class RecommendResult:
    df_movie_rating_pred: DFMovieRatingPred | None
    df_user_movies_pred: DFUserMoviesPred


@dataclass(frozen=True)
class Metrics:
    rmse: float | None
    precision_at_k: float
    recall_at_k: float

    def __repr__(self):
        return f"rmse={self.rmse}, Precision@K={self.precision_at_k}, Recall@K={self.recall_at_k}"
