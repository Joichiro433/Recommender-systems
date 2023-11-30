from collections import defaultdict

from rich import print
import numpy as np
import polars as pl
from lightfm import LightFM
from lightfm.data import Dataset as LFMDataset

from utils.types import RecommendResult, Dataset, DFData, DFUserMovies, DFMovieRatingPred, DFUserMoviesPred
from recommenders.base_recommend import BaseRecommender


class MFRecommender(BaseRecommender):
    """
    A recommender system based on Matrix Factorization (MF).

    Attributes
    ----------
    dataset : Dataset
        The dataset used for the recommender system.

    Methods
    -------
    recommend() -> RecommendResult
        Generates recommendations using Matrix Factorization.
    """
    def __init__(self, dataset: Dataset) -> None:
        super().__init__(dataset)

    def recommend(self) -> RecommendResult:
        """
        Generates recommendations using Matrix Factorization.

        Returns
        -------
        RecommendResult
            The result of the recommendation, including predicted movie ratings and predicted movies for each user.
        """
        df_train: DFData = self.dataset.df_train
        df_test: DFData = self.dataset.df_test
        df_user_movies_test: DFUserMovies = self.dataset.df_user_movies_test

        unique_user_ids: pl.Series = df_train['user_id'].unique()
        unique_movie_ids: pl.Series = df_train['movie_id'].unique()

        lfm_dataset: LFMDataset = LFMDataset()
        lfm_dataset.fit(users=unique_user_ids, items=unique_movie_ids)

        df_train_high_rating: pl.DataFrame = df_train.filter(pl.col('rating') >= 4.0)
        user_movie_pairs: np.ndarray = df_train_high_rating[['user_id', 'movie_id']].unique().to_numpy()

        train_interactions, _ = lfm_dataset.build_interactions(user_movie_pairs)
        user_id_map, user_feature_map, item_id_map, item_feature_map = lfm_dataset.mapping()

        model = LightFM(no_components=10, loss='warp', random_state=42)
        model.fit(interactions=train_interactions, epochs=100)

        df_user_evaluated_movies: pl.DataFrame = (
            df_train
            .group_by('user_id')
            .agg(movie_id_evaluated=pl.col('movie_id'))
        )

        user_movies_pred: dict[str, list] = defaultdict(list)
        for user_id, evaluated_movie_ids in df_user_evaluated_movies.to_numpy():
            unevaluated_movie_ids: list[int] = list(set(unique_movie_ids) - set(evaluated_movie_ids))
            user_index: int = user_id_map[user_id]
            unevaluated_movie_indices: list[int] = [item_id_map[mid] for mid in unevaluated_movie_ids]
            scores: list[float] = model.predict(user_index, unevaluated_movie_indices)
            df_score: pl.DataFrame = pl.DataFrame({'movie_id': unevaluated_movie_ids, 'score': scores})
            recommend_movies: list[int] = (
                df_score
                .sort('score', descending=True)
                .head(10)
                .get_column('movie_id')
                .to_list()
            )
            user_movies_pred['user_id'].append(user_id)
            user_movies_pred['movie_id_pred'].append(recommend_movies)
        
        df_user_movies_pred: DFUserMoviesPred = (
            df_user_movies_test
            .join(pl.DataFrame(user_movies_pred), on='user_id', how='left')
        )
        # 評価値の予測は難しいため、rmseの評価は行わない（便宜上、テストデータの予測値をそのまま返す）
        df_movie_rating_pred: DFMovieRatingPred = (
            df_test
            .with_columns(rating_pred=pl.col('rating'))
        )
        return RecommendResult(df_movie_rating_pred=df_movie_rating_pred, df_user_movies_pred=df_user_movies_pred)
