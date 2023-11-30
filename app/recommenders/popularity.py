from collections import defaultdict

from rich import print
import polars as pl

from utils.types import RecommendResult, Dataset, DFData, DFUserMovies, DFMovieRatingPred, DFUserMoviesPred
from recommenders.base_recommend import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """
    A class used to recommend movies based on their popularity.

    The popularity of a movie is determined by the average of its ratings. 
    Only movies with a minimum number of ratings are considered.

    Attributes
    ----------
    dataset : Dataset
        The dataset containing the movie ratings.
    minimum_num_rating : int
        The minimum number of ratings a movie must have to be considered.

    Methods
    -------
    recommend() -> RecommendResult
        Recommends movies for each user in the dataset. The recommendations are based on the popularity of the movies.
    """
    def __init__(self, dataset: Dataset, minimum_num_rating: int = 200) -> None:
        super().__init__(dataset=dataset)
        self.minimum_num_rating: int = minimum_num_rating
        
    def recommend(self) -> RecommendResult:
        """
        Recommends movies for each user in the dataset. The recommendations are based on the popularity of the movies.

        Returns
        -------
        RecommendResult
            The predicted movie ratings and the top 10 recommended movies for each user.
        """
        df_movie_rating_pred: DFMovieRatingPred = self._pred_rating()
        df_user_movies_pred: DFUserMoviesPred = self._pred_recommend_movies()
        return RecommendResult(df_movie_rating_pred=df_movie_rating_pred, df_user_movies_pred=df_user_movies_pred)
    
    def _pred_rating(self) -> DFMovieRatingPred:
        """
        Predicts the rating for each movie in the test set based on the average rating of the movie in the training set.

        Returns
        -------
        DFMovieRatingPred
            The DataFrame containing the true and predicted ratings for each movie in the test set.
        """
        df_train: DFData = self.dataset.df_train
        df_test: DFData = self.dataset.df_test

        df_movie_rating_avg: pl.DataFrame = (
            df_train
            .group_by('movie_id')
            .agg(pl.col('rating').mean())
        )
        df_movie_rating_pred: DFMovieRatingPred = (
            df_test
            .select(['user_id', 'movie_id', 'rating'])
            .join(df_movie_rating_avg, on='movie_id', how='left', suffix='_pred')
            .fill_null(0)  # df_trainに存在しない映画のratingは 0 にする
        )
        return df_movie_rating_pred        

    def _pred_recommend_movies(self) -> DFUserMoviesPred:
        """
        Predicts the top 10 movies for each user based on their popularity.

        Returns
        -------
        DFUserMoviesPred
            The DataFrame containing the top 10 recommended movies for each user.
        """
        df_train: DFData = self.dataset.df_train
        df_user_movies_test: DFUserMovies = self.dataset.df_user_movies_test

        movies_sorted_by_rating: list[int] = (
            df_train
            .group_by('movie_id')
            .agg(
                rating_count=pl.col('rating').count(), 
                rating_mean=pl.col('rating').mean(),
            )
            .filter(pl.col('rating_count') >= self.minimum_num_rating)  # 評価件数が少ないとノイズが大きいため、minimum_num_rating件以上評価がある映画に絞る
            .sort('rating_mean', descending=True)
            .get_column('movie_id')
            .to_list()
        )
        df_user_watched_movie: pl.DataFrame = (
            df_train
            .groupby('user_id')
            .agg(user_watched_movies=pl.col('movie_id'))
        )
        # 各ユーザに対するおすすめ映画は、そのユーザがまだ評価していない映画の中から評価値が高いもの10作品とする
        user_movies_pred: dict[str, list] = defaultdict(list)
        for user_id, watched_movies in df_user_watched_movie.to_numpy():
            recommend_movies: list[int] = []
            for movie_id in movies_sorted_by_rating:
                if movie_id not in watched_movies:
                    recommend_movies.append(movie_id)
                if len(recommend_movies) >= 10:
                    break
            user_movies_pred['user_id'].append(user_id)
            user_movies_pred['movie_id_pred'].append(recommend_movies)
        
        df_user_movies_pred: DFUserMoviesPred = (
            df_user_movies_test
            .join(pl.DataFrame(user_movies_pred), on='user_id', how='left')
        )
        return df_user_movies_pred
