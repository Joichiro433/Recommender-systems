from abc import ABC, abstractclassmethod

from rich import print

from utils.metrics import MetricCalculator
from utils.types import Dataset, RecommendResult, Metrics


class BaseRecommender(ABC):
    """
    Abstract base class for a recommender system.

    Attributes
    ----------
    dataset : Dataset
        The dataset used for the recommender system.

    Methods
    -------
    recommend() -> RecommendResult
        Abstract method to generate movie recommendations for each user in the dataset.
    run_sample(k: int = 10) -> None
        Runs a sample recommendation, calculates metrics, and prints the results.
    """
    def __init__(self, dataset: Dataset) -> None:
        self.dataset: Dataset = dataset

    @abstractclassmethod
    def recommend(self) -> RecommendResult:
        """
        Abstract method to generate movie recommendations for each user in the dataset.

        Raises
        ------
        NotImplementedError
            This method must be implemented in a subclass.
        """
        raise NotImplementedError()
    
    def run_sample(self, k: int = 10) -> None:
        """
        Runs a sample recommendation, calculates metrics, and prints the results.

        Parameters
        ----------
        k : int, optional
            The number of top recommendations to consider, by default 10.
        """
        recommend_result: RecommendResult = self.recommend()

        metrics: Metrics = MetricCalculator.calc(
            df_movie_rating_pred=recommend_result.df_movie_rating_pred,
            df_user_movies_pred=recommend_result.df_user_movies_pred,
            k=k,
        )

        print(recommend_result.df_movie_rating_pred.sort('user_id'))
        print(recommend_result.df_user_movies_pred.sort('user_id'))
        print(metrics)
