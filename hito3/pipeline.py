from typing import List, Dict, Tuple
from pprint import pformat
from collections import defaultdict, OrderedDict
import requests

import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class InfoDisplayer(BaseEstimator, TransformerMixin):
    """Display info about a DataFrame."""

    def __init__(self, name: str):
        self.name = name

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, list):
            message = f"[{self.name}]"
            for df in X:
                " | ".join([message, f"shape: {df.shape}"])
            print(message)
        else:
            print(f"[{self.name}] shape: {X.shape}")
        return X


class DropColumns(BaseEstimator, TransformerMixin):
    """Drop columns from a DataFrame."""

    def __init__(self, columns: List[str], errors: str = "raise"):
        self.columns = columns
        self.errors = errors

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns, errors=self.errors)

    def __repr__(self):
        return f"DropColumns(columns={pformat(self.columns)})"


class DropHighNAPercentage(BaseEstimator, TransformerMixin):
    """Drop columns with a percentage of NA values higher than a threshold."""

    def __init__(
        self,
        exclude: List[str],
        na_threshold: float = 0.26,
    ):
        if na_threshold < 0 or na_threshold > 1:
            raise ValueError("na_threshold must be between 0 and 1")
        self.na_threshold = na_threshold
        self.exclude = exclude

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        vacios = X.isnull().sum()
        vacios = (vacios[vacios > 0] / X.shape[0]) * 100
        vacios = vacios.astype(int)
        vacios = vacios.sort_values(ascending=False)
        indices_vacios = vacios[vacios > int(self.na_threshold * 100)].index
        return X.drop(columns=indices_vacios.difference(self.exclude))

    def __repr__(self):
        return f"DropHighNAPercentage(na_threshold={self.na_threshold})"


def get_uf_value():
    url = "https://www.mindicador.cl/api"
    response = requests.get(url)
    return response.json()["uf"]["valor"]


class NormalizeCurrency(BaseEstimator, TransformerMixin):
    """Preprocess Tipo Moneda column."""

    def __init__(self, columns: List[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X["Tipo Moneda"] = X["Tipo Moneda"].str.strip().str.capitalize()

        uf_value = get_uf_value()
        uf_index_mask = X["Tipo Moneda"] == "Uf"
        X.loc[uf_index_mask, self.columns] = (
            X.loc[uf_index_mask, self.columns] * uf_value
        )
        X = X.drop(columns=["Tipo Moneda"])

        return X


def map_column_to_ordinal(
    df: pd.DataFrame,
    column: str,
    mapping: Dict[str, int],
) -> pd.DataFrame:
    df = df.copy()
    df[column] = df[column].map(mapping)
    return df


class OrdinalColumnMapper(BaseEstimator, TransformerMixin):
    """Map column values to new values."""

    def __init__(
        self,
        columns: List[str],
        mappings: List[Dict[str, int]],
    ):
        self.columns = columns
        self.mappings = mappings

        self.inverse_mappings = {
            columns[i]: defaultdict(list) for i in range(len(columns))
        }
        for i, mapping in enumerate(mappings):
            for k, v in mapping.items():
                self.inverse_mappings[columns[i]][v].append(k)

        for column, mapping in self.inverse_mappings.items():
            for k, v in mapping.items():
                self.inverse_mappings[column][k] = " | ".join(v)

        for column, mapping in self.inverse_mappings.items():
            self.inverse_mappings[column] = dict(mapping)

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for column in self.columns:
            X[column] = X[column].astype("category")

        for column, mapping in zip(self.columns, self.mappings):
            X = map_column_to_ordinal(X, column, mapping)
        return X

    def __repr__(self):
        return (
            f"OrdinalColumnMapper(columns={pformat(self.columns)},"
            f"mappings={pformat(self.mappings)})"
        )


class DataframeOneHotEncoder(BaseEstimator, TransformerMixin):
    """One hot encode a column."""

    def __init__(
        self,
        columns: List[str],
        min_frequency: int | None = None,
        max_categories: int | None = None,
    ):
        self.columns = columns
        self.encoder = OneHotEncoder(
            sparse_output=False,
            min_frequency=min_frequency,
            max_categories=max_categories,
        )
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    def fit(self, X: pd.DataFrame, y=None):
        self.encoder.fit(X[self.columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = self.encoder.transform(X[self.columns])
        onehot_df = pd.DataFrame(
            data=data,
            columns=self.encoder.get_feature_names_out(),
        ).astype("category")
        X = X.drop(columns=self.columns)
        X = pd.concat([X, onehot_df], axis=1)
        return X


class NanInputer(BaseEstimator, TransformerMixin):
    """Inpute NA values."""

    def __init__(
        self,
        columns: List[str] | str = "auto",
        n_neighbors: int = 5,
    ):
        self.columns = columns
        # self.inputer = KNNImputer(n_neighbors=n_neighbors).set_output(
        #     transform="pandas"
        # )
        self.inputer = SimpleImputer(strategy="most_frequent").set_output(
            transform="pandas"
        )
        self.n_neighbors = n_neighbors

    def fit(self, X: pd.DataFrame, y=None):
        if self.columns == "auto":
            self.columns = X.isnull().any().index
        self.inputer.fit(X[self.columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X[self.columns] = self.inputer.transform(X[self.columns])
        return X


class MultiDataFramePipeline(BaseEstimator, TransformerMixin):
    """Apply a list of transformers to a list of DataFrames. Each transformer
    is applied to a single DataFrame. The outputs are returned in a list."""

    def __init__(self, transformers: List[Tuple[str, TransformerMixin]]):
        self.transformers = OrderedDict(transformers)

    def fit(self, X: List[pd.DataFrame], y=None):
        if len(X) != len(self.transformers):
            raise ValueError(
                f"Expected {len(self.transformers)} DataFrames, got {len(X)}"
            )

        for df, transformer in zip(X, self.transformers.values()):
            print(f"Fitting {transformer.__class__.__name__}...")
            transformer.fit(df)
        return self

    def transform(self, X: List[pd.DataFrame]) -> List[pd.DataFrame]:
        if len(X) != len(self.transformers):
            raise ValueError(
                f"Expected {len(self.transformers)} DataFrames, got {len(X)}"
            )

        results = []
        for df, transformer in zip(X, self.transformers.values()):
            results.append(transformer.transform(df))

        print(f"Returning {len(results)} DataFrames.")
        return results

    def _repr_html_(self):
        reprs = [
            f"<li>{transformer.__class__.__name__}(parameters={transformer.get_params()})</li>"
            for transformer in self.transformers.values()
        ]
        return f"<ul>{''.join(reprs)}</ul>"


def merge_dataframes(
    target_df: pd.DataFrame,
    df_to_merge: pd.DataFrame,
    merge_cols: List[str],
) -> pd.DataFrame:
    """
    Merge two dataframes on the specified columns,
    conserving the index of the target dataframe
    Args:
        target_df (pd.DataFrame): dataframe to merge with
        df_to_merge (pd.DataFrame): dataframe to merge
        merge_cols (List[str]): columns to merge on
    Returns:
        pd.DataFrame: merged dataframe
    """
    target_df = target_df.copy()

    target_df["index"] = target_df.index
    df_merged = target_df.merge(df_to_merge, on=merge_cols, how="inner")
    df_merged = df_merged.set_index("index")
    df_merged.index.name = None
    return df_merged


class MultiDataFrameMerger(BaseEstimator, TransformerMixin):
    """Merge a list of DataFrames."""

    def __init__(self, columns: List[List[str]] | List[str]):
        """Initialize the MultiDataFrameMerger.

        Parameters:
        columns (List[List[str]] | List[str]): A list of lists of columns to
        merge. Each list is used to merge a pair of DataFrames consecutively.
        If the list has only one element, it's used on every step.
        """

        self.columns = columns

    def fit(self, X: List[pd.DataFrame], y=None):
        print(f"Merging {len(X)} DataFrames into one...")

        if isinstance(self.columns[0], str):
            self.columns = [self.columns] * (len(X) - 1)

        if len(X) != len(self.columns) + 1:
            raise ValueError(
                f"Expected {len(self.columns) + 1} DataFrames, got {len(X)}"
            )

        return self

    def transform(self, X: List[pd.DataFrame]) -> pd.DataFrame:
        if len(X) != len(self.columns) + 1:
            raise ValueError(
                f"Expected {len(self.columns) + 1} DataFrames, got {len(X)}"
            )

        df_merged = X[0]
        for df, columns in zip(X[1:], self.columns):
            df_merged = merge_dataframes(df_merged, df, columns)
        return df_merged


class KMeansClusterer(BaseEstimator, TransformerMixin):
    """Perform KMeans clustering with multiple values and store scores."""

    def __init__(
        self,
        n_clusters: List[int],
        columns: List[str] | None = None,
        random_state: int = 42,
        progress_bar: bool = False,
    ):
        self.columns = columns
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.progress_bar = progress_bar

        self.clusters_data = {}
        self.results_df = None

    def fit(self, X: pd.DataFrame, y=None):
        self.columns = self.columns or X.columns

        iterator = self.n_clusters
        if self.progress_bar:
            iterator = tqdm(self.n_clusters, desc="KMeans")

        for n_clusters in iterator:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init="auto",
            )
            kmeans.fit(X[self.columns])

            self.clusters_data[n_clusters] = {
                "model": kmeans,
                "silhouette_score": silhouette_score(
                    X[self.columns],
                    kmeans.labels_,
                ),
            }

            self.results_df = pd.DataFrame.from_dict(
                self.clusters_data,
                orient="index",
            ).sort_index()
            self.results_df.index.name = "n_clusters"
            self.results_df["inertia"] = self.results_df["model"].apply(
                lambda x: x.inertia_
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        best_n_clusters = max(
            self.clusters_data,
            key=lambda n_clusters: self.clusters_data[n_clusters]["silhouette_score"],
        )
        kmeans = self.clusters_data[best_n_clusters]["model"]
        X["cluster"] = kmeans.labels_
        return X

    def get_labels(self, n_clusters: int) -> pd.Series:
        if n_clusters not in self.clusters_data:
            raise ValueError(f"n_clusters={n_clusters} not found")

        kmeans = self.clusters_data[n_clusters]["model"]
        return kmeans.labels_

    def plot_elbow(self):
        self.results_df.plot(x="n_clusters", y="inertia", kind="line")

    def __repr__(self):
        return f"KMeansClusterer(n_clusters={pformat(self.n_clusters)})"
