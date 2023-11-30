from typing import TypeAlias
from collections import defaultdict, Counter

from rich import print
import numpy as np
from nptyping import DataFrame, Structure as S
import pandas as pd
import polars as pl
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

from utils.types import RecommendResult, Dataset, DFMovieRatingPred, DFUserMoviesPred
from recommenders.base_recommend import BaseRecommender


DFLift: TypeAlias = pl.DataFrame | DataFrame[S[
    """
    antecedents: List[Int],
    consequents: List[Int],
    lift: Float
    """
]]


class AssociationRecommender(BaseRecommender):
    """
    A class used to recommend movies based on association rules.

    Attributes
    ----------
    dataset : Dataset
        The dataset used for recommendation.
    min_support : float
        The minimum support for the apriori algorithm.
    min_threshold : float
        The minimum threshold for the association rules.

    Methods
    -------
    recommend() -> RecommendResult
        Recommends movies for each user in the dataset.
    """
    def __init__(
            self, 
            dataset: Dataset, 
            min_support: float = 0.1, 
            min_threshold: float = 1.0,
        ) -> None:
        super().__init__(dataset=dataset)
        self.min_support: float = min_support
        self.min_threshold: float = min_threshold

    def recommend(self) -> RecommendResult:
        """
        Recommends movies for each user in the dataset.

        Returns
        -------
        RecommendResult
            The recommended movies for each user in the dataset.
        """
        # ユーザーが評価した映画一覧
        df_user_evaluated_movies: pl.DataFrame = (
            self.dataset.df_train
            .group_by('user_id')
            .agg('movie_id')
        )
        # 評価値が4以上の中で、直近評価した映画5件をユーザーごとに取得
        df_train_latest_high_rating_movies: pl.DataFrame = (
            self.dataset.df_train
            .filter(pl.col('rating') >= 4)
            .with_columns(
                pl.col('timestamp')
                .rank(method='ordinal', descending=True)  # 直近評価した映画から順番を付与
                .over('user_id')
                .alias('rating_order')
            )
            .filter(pl.col('rating_order') <= 5)
            .group_by('user_id')
            .agg('movie_id')
        )
        df_movies: pl.DataFrame = (
            df_train_latest_high_rating_movies
            .join(df_user_evaluated_movies, on='user_id', how='left', suffix='_evaluated')
        )

        user_movies_pred: dict[str, list] = defaultdict(list)
        df_lift: DFLift = self._calc_lift()
        for user_id, movies, evaluated_movies in df_movies.to_numpy():
            consequent_movies: list[int] = (
                df_lift
                .with_columns(
                    pl.col('antecedents')
                    .list.set_intersection(list(movies))
                    .list.len()
                    .alias('num_match')
                )
                .filter(pl.col('num_match') >= 1)
                .get_column('consequents').list.explode().to_list()
            )
            counter: Counter = Counter(consequent_movies)
            recommend_movies: list[int] = []
            for movie_id, _ in counter.most_common():
                if movie_id not in evaluated_movies:
                    recommend_movies.append(movie_id)
                # 推薦リストが10本になったら終了する
                if len(recommend_movies) >= 10:
                    break
            user_movies_pred['user_id'].append(user_id)
            user_movies_pred['movie_id_pred'].append(recommend_movies)

        df_user_movies_pred: DFUserMoviesPred = (
            self.dataset.df_user_movies_test
            .join(pl.DataFrame(user_movies_pred), on='user_id', how='left')
        )
        # アソシエーションルールでは評価値の予測は難しいため、rmseの評価は行わない（便宜上、テストデータの予測値をそのまま返す）
        df_movie_rating_pred: DFMovieRatingPred = (
            self.dataset.df_test
            .with_columns(rating_pred=pl.col('rating'))
        )
        return RecommendResult(df_movie_rating_pred=df_movie_rating_pred, df_user_movies_pred=df_user_movies_pred)

    def _calc_lift(self) -> DFLift:
        """
        Calculates the lift for each pair of movies in the dataset.

        Returns
        -------
        DFLift
            The lift for each pair of movies in the dataset.
        """
        df_user_movie_matrix: pd.DataFrame = (
            self.dataset.df_train
            .select(['user_id', 'movie_id', 'rating'])
            .with_columns(  # 4以上の評価値はTrue, 4未満の評価値と欠損値はFalseにする
                pl.when(pl.col('rating') >= 4)
                .then(True)
                .otherwise(False)
                .alias('rating')
            )
            .to_pandas()  # ライブラリが読み込める形式に整形
            .pivot(index='user_id', columns='movie_id', values='rating')
            .fillna(False)
        )
        # 支持度の計算
        df_freq_movies: pd.DataFrame = apriori(
            df=df_user_movie_matrix,
            min_support=self.min_support, 
            use_colnames=True
        )
        # リフト値の計算
        df_lift: pd.DataFrame = association_rules(
            df=df_freq_movies, 
            metric="lift", 
            min_threshold=self.min_threshold
        )[['antecedents', 'consequents', 'lift']]
        # antecedents, consequents は type: frozenset となっているが、これはpolarsで読み込むことができない
        # 一旦 type: set に変換した後、polars で読み込むことで type: list とすることができる
        @np.vectorize
        def to_set(value):
            return set(value)

        df_lift['antecedents'] = to_set(df_lift['antecedents'])
        df_lift['consequents'] = to_set(df_lift['consequents'])
        df_lift: DFLift = pl.from_pandas(df_lift)
        return df_lift.sort('lift', descending=True)
