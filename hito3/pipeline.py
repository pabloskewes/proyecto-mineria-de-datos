from typing import List, Dict
from pprint import pformat
from collections import defaultdict
import requests

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer


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
        return f"DropColumns(columns_to_drop={pformat(self.columns)})"


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
        for column in self.columns:
            X[column] = X[column].astype("category")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for column, mapping in zip(self.columns, self.mappings):
            X = map_column_to_ordinal(X, column, mapping)
        return X

    def __repr__(self):
        return (
            f"ColumnMapper(columns={pformat(self.columns)},"
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
        self.inputer = KNNImputer(n_neighbors=n_neighbors).set_output(
            transform="pandas"
        )
        self.n_neighbors = n_neighbors

    def fit(self, X: pd.DataFrame, y=None):
        if self.columns == "auto":
            self.columns = X.isnull().any().index
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X[self.columns] = self.inputer.fit_transform(X[self.columns])
        return X


class MultiDataFramePipeline:
    """Apply a list of transformers to a list of DataFrames. Each transformer
    is applied to a single DataFrame. The outputs are returned in a list."""

    def __init__(self, transformers: List[TransformerMixin]):
        self.transformers = transformers

    def fit(self, X: List[pd.DataFrame], y=None):
        for transformer in self.transformers:
            transformer.fit(X)
        return self

    def transform(self, X: List[pd.DataFrame]) -> List[pd.DataFrame]:
        return [
            transformer.transform(x) for transformer, x in zip(X, self.transformers)
        ]

    def __repr__(self):
        return f"MultiDataFramePipeline(transformers={pformat(self.transformers)})"
