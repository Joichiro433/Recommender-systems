from typing import Annotated, TypeAlias

import pretty_errors
import typer

from recommenders.base_recommend import BaseRecommender
from recommenders.popularity import PopularityRecommender
from recommenders.association import AssociationRecommender
from recommenders.knn_collab_filtering import KNNCollabFilteringRecommender
from recommenders.random_forest import RandomForestRecommender
from recommenders.als import AlternatingLeastSquaresRecommender
from recommenders.mf import MFRecommender
from recommenders.lightfm import LightFMRecommender
from utils.data_loader import DataLoader


RECOMMENDER_MAP = {
    'popularity': PopularityRecommender,
    'association': AssociationRecommender,
    'knn_collab_filtering': KNNCollabFilteringRecommender,
    'random_forest': RandomForestRecommender,
    'als': AlternatingLeastSquaresRecommender,
    'mf': MFRecommender,
    'lightfm': LightFMRecommender,
}

def validate_recommender(recommender_type: str):
    if recommender_type not in RECOMMENDER_MAP:
        raise typer.BadParameter(f'Recommender must be one of {list(RECOMMENDER_MAP.keys())}')
    return recommender_type

RecommenderArg: TypeAlias = Annotated[str, typer.Option('--recommender', '-r', callback=validate_recommender)]

def main(recommender_type: RecommenderArg = 'mf'):
    movielens = DataLoader().load()
    recommender: BaseRecommender = RECOMMENDER_MAP[recommender_type](dataset=movielens)
    recommender.run_sample()


if __name__ == '__main__':
    typer.run(main)
