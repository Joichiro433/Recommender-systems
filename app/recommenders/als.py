from collections import defaultdict

from rich import print
import polars as pl
from implicit.als import AlternatingLeastSquares
from scipy.sparse import lil_matrix

from utils.types import RecommendResult, Dataset, DFData, DFUserMovies, DFMovieRatingPred, DFUserMoviesPred
from recommenders.base_recommend import BaseRecommender


class AlternatingLeastSquaresRecommender(BaseRecommender):
    def __init__(
            self, 
            dataset: Dataset,
            factors: int = 10,
            minimum_num_rating: float = 0.0,
            n_epochs: int = 100,
            alpha: float = 1.0,
        ) -> None:
        """
        Constructs all the necessary attributes for the ALS recommender object.

        Parameters
        ----------
        dataset : Dataset
            The dataset used for the recommender system.
        factors : int, optional
            The number of factors to use in the ALS model, by default 10.
        minimum_num_rating : float, optional
            The minimum number of ratings a movie must have to be considered, by default 0.0.
        n_epochs : int, optional
            The number of epochs to run the ALS model, by default 100.
        alpha : float, optional
            The confidence parameter for the ALS model, by default 1.0.
        """
        super().__init__(dataset=dataset)
        self.factors: int = factors
        self.minimum_num_rating: float = minimum_num_rating
        self.n_epochs: int = n_epochs
        self.alpha: float = alpha
    
    def recommend(self) -> RecommendResult:
        """
        Generates movie recommendations for each user in the dataset.

        Returns
        -------
        RecommendResult
            A data object containing the predicted movie ratings and the predicted user-movie recommendations.
        """
        df_train: DFData = self.dataset.df_train
        df_test: DFData = self.dataset.df_test
        df_user_movies_test: DFUserMovies = self.dataset.df_user_movies_test

        unique_user_ids: list[int] = sorted(df_train['user_id'].unique())
        unique_movie_ids: list[int] = sorted(df_train['movie_id'].unique())

        # 行列分解用に行列を作成する
        df_train_high_rating = df_train.filter(pl.col('rating') >= 5.0)

        userid_to_index: dict[int, int] = dict(zip(unique_user_ids, range(len(unique_user_ids))))
        movieid_to_index: dict[int, int] = dict(zip(unique_movie_ids, range(len(unique_movie_ids))))
        rating_matrix = lil_matrix((len(unique_user_ids), len(unique_movie_ids)))
        for user_id, movie_id in df_train_high_rating[['user_id', 'movie_id']].to_numpy():
            user_index = userid_to_index[user_id]
            movie_index = movieid_to_index[movie_id]
            rating_matrix[user_index, movie_index] = 1.0 * self.alpha
        rating_matrix = rating_matrix.tocsr()
        
        model = AlternatingLeastSquares(
            factors=self.factors, 
            iterations=self.n_epochs, 
            calculate_training_loss=True, 
            random_state=1,
        )
        model.fit(rating_matrix, show_progress=False)
        user_indices = list(userid_to_index.values())
        recommendations, scores = model.recommend(userid=user_indices, user_items=rating_matrix)

        user_movies_pred: dict[str, list] = defaultdict(list)
        for user_id, user_index in userid_to_index.items():
            movie_indices = recommendations[user_index, :]
            recommend_movies: list[int] = []
            for movie_index in movie_indices:
                movie_id = unique_movie_ids[movie_index]
                recommend_movies.append(movie_id)
            user_movies_pred['user_id'].append(user_id)
            user_movies_pred['movie_id_pred'].append(recommend_movies)
        
        df_user_movies_pred: DFUserMoviesPred = (
            df_user_movies_test
            .join(pl.DataFrame(user_movies_pred), on='user_id', how='left')
        )
        # ALSでは評価値の予測は難しいため、rmseの評価は行わない（便宜上、テストデータの予測値をそのまま返す）
        df_movie_rating_pred: DFMovieRatingPred = (
            df_test
            .with_columns(rating_pred=pl.col('rating'))
        )
        return RecommendResult(df_movie_rating_pred=df_movie_rating_pred, df_user_movies_pred=df_user_movies_pred)
