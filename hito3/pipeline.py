from typing import List, Tuple, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


COLUMNS_TO_DROP = [
    "Nombre de la Sede",
    "Orden Geográfico de la Región (Norte aSur)",
    "Mención o Especialidad",
    "idgenerocarrera",
    "Cód. Campus",
    "Cód. Sede",
    "Códgo SIES",
    "Máximo Puntaje (promedio matemáticas y lenguaje)",
    "Máximo Puntaje NEM",
    "Máximo Puntaje Ranking",
    "Mínimo Puntaje (promedio matemáticas y lenguaje)",
    "Mínimo Puntaje NEM",
    "Mínimo Puntaje Ranking",
]


class DropColumns(BaseEstimator, TransformerMixin):
    """Drop columns from a DataFrame."""

    def __init__(self, columns_to_drop: List[str] = COLUMNS_TO_DROP):
        self.columns_to_drop = columns_to_drop

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.columns_to_drop)


class DropHighNAPercentage(BaseEstimator, TransformerMixin):
    """Drop columns with a percentage of NA values higher than a threshold."""

    def __init__(self, na_threshold: float = 0.24):
        if na_threshold < 0 or na_threshold > 1:
            raise ValueError("na_threshold must be between 0 and 1")
        self.na_threshold = na_threshold

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        vacios = X.isnull().sum()
        vacios = (vacios[vacios > 0] / X.shape[0]) * 100
        vacios = vacios.astype(int)
        vacios = vacios.sort_values(ascending=False)
        indices_vacios = vacios[vacios > int(self.na_threshold * 100)].index
        return X.drop(columns=indices_vacios)


class PreprocessTipoMoneda(BaseEstimator, TransformerMixin):
    """Preprocess Tipo Moneda column."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X["Tipo Moneda"] = X["Tipo Moneda"].str.strip().str.capitalize()
        return X
