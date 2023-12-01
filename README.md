# Movie Recommender System

This project implements a movie recommender system using various recommendation algorithms. The system is designed to recommend movies to users based on their past ratings. The implementation is based on the book: [風間正弘、飯塚洸二郎、松村優也著『推薦システム実践入門』（オライリー・ジャパン、ISBN978-4-87311-966-3）](https://www.oreilly.co.jp/books/9784873119663/)

### Overview

The recommender system is implemented in Python and uses several libraries including typer, pretty_errors, and polars. The system supports multiple recommendation algorithms including:

- Popularity-based Recommender
- Association Rule Recommender
- K-Nearest Neighbors Collaborative Filtering Recommender
- Random Forest Recommender
- Alternating Least Squares Recommender
- Matrix Factorization Recommender
- LightFM Recommender

Each recommender is implemented as a separate class that inherits from the BaseRecommender abstract base class.

### Prerequisites

This project requires Python 3.10. Please ensure that you have the correct Python version installed before proceeding.

### Installation

This project uses Poetry for dependency management. To install the project dependencies, first install Poetry:

```sh
$ pip install poetry
```

Then, navigate to the project directory and run:

```sh
$ poetry install
```

This will install all the necessary dependencies as specified in the pyproject.toml and poetry.lock files.

### Data

Before using the recommender system, you need to download and preprocess the dataset. This can be done by running the `download_dataset.py` script:

```sh
$ python download_dataset.py
```

This script downloads the MovieLens dataset, unzips it, and converts the data files to Parquet format for easier processing.

The system uses the MovieLens dataset. The data is loaded and preprocessed using the DataLoader class. The data is split into training and testing sets, and the ratings are preprocessed to be suitable for each recommender.

### Usage

The main entry point of the application is `main.py`. The recommender to use can be specified using the `--recommender` or `-r` command line option. The available options are `{popularity, association, knn_collab_filtering, random_forest, als, mf, lightfm}`. For example, to use the matrix factorization recommender, you would run:

```sh
$ python main.py --recommender mf
```

If no recommender is specified, the matrix factorization recommender is used by default.

### Metrics

The system calculates several metrics to evaluate the performance of the recommenders, including RMSE (Root Mean Square Error), Recall@K and Precision@K.

### Customization

You can customize the behavior of the recommenders by modifying the parameters in the respective recommender classes. For example, you can change the number of factors used in the ALS recommender by modifying the factors attribute of the AlternatingLeastSquaresRecommender class.

### Future Work

Future improvements could include adding more recommenders, improving the performance of the existing recommenders, and adding more metrics for evaluation.

