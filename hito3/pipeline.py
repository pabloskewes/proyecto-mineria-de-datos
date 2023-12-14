from typing import List, Dict
from pprint import pformat
from collections import defaultdict
import requests


import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

COLUMNS_TO_DROP = [
    "Nombre de la Sede",
    "Orden Geográfico de la Región (Norte aSur)",
    "Mención o Especialidad",
    "idgenerocarrera",
    # "Cód. Campus",
    # "Cód. Sede",
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

    def __repr__(self):
        return f"DropColumns(columns_to_drop={pformat(self.columns_to_drop)})"


EXCLUDE_COLUMNS_OF_DROPHIGHNA = ["Nombre del Campus"]


class DropHighNAPercentage(BaseEstimator, TransformerMixin):
    """Drop columns with a percentage of NA values higher than a threshold."""

    def __init__(
        self,
        na_threshold: float = 0.26,
        exclude: List[str] = EXCLUDE_COLUMNS_OF_DROPHIGHNA,
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


CURRENCY_COLUMNS = [
    "Valor de matrícula",
    "Valor de arancel",
    "Valor del Título",
]


class NormalizeCurrency(BaseEstimator, TransformerMixin):
    """Preprocess Tipo Moneda column."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X["Tipo Moneda"] = X["Tipo Moneda"].str.strip().str.capitalize()

        uf_value = get_uf_value()
        uf_index_mask = X["Tipo Moneda"] == "Uf"
        X.loc[uf_index_mask, CURRENCY_COLUMNS] = (
            X.loc[uf_index_mask, CURRENCY_COLUMNS] * uf_value
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


m_class1 = {
    "(a) Universidades CRUCH": 0,
    "(b) Universidades Privadas": 1,
    "(c) Institutos Profesionales": 2,
    "(d) Centros de Formación Técnica": 3,
    "(e) Centros de Formación Técnica Estatales": 3,
    "(f) F.F.A.A.": 4,
}

m_class2 = {
    "(a) Universidades Estatales CRUCH": 0,
    "(b) Universidades Privadas CRUCH": 1,
    "(c) Univ. Privadas Adscritas SUA": 2,
    "(d) Universidades Privadas": 3,
    "(e) Institutos Profesionales": 4,
    "(f) Centros de Formación Técnica": 5,
    "(g) Centros de Formación Técnica statales": 5,
    "(h) F.F.A.A.": 6,
}

m_class3 = {"(a) Acreditada": 0, "(b) No Acreditada": 1}

m_class4 = {
    "(a) Autónoma": 0,
    "(b) Licenciamiento": 1,
    "(c) Examinación": 2,
    "(d) Supervisión": 3,
    "(e) F.F.A.A.": 4,
    "(e) Cerrada": 5,
}

m_class5 = {"(a) Adscritas a Gratuidad": 0, "(b) No Adscritas/No Aplica": 1}

m_class6 = {
    "(a) Subsistema Universitario": 0,
    "(b) Subsistema Técnico Profesional": 1,
    "(c) No adscrito": 2,
    "(d) F.F.A.A.": 3,
}

m_inst = {
    "Univ.": 0,
    "I.P.": 1,
    "C.F.T.": 2,
    "F.F.A.A.": 3,
}

m_grado = {
    "Ingeniero": 0,
    "Licencia": 1,
    "Bachiller": 2,
    "Preparador": 3,
    "No aplica": 4,
    "Desconocido": 5,
}


COLUMNS_TO_MAP = [f"Clasificación{i}" for i in range(1, 7)]
COLUMNS_TO_MAP.extend(
    [
        "Tipo Institución",
        "Grado Académico",
    ]
)
MAPPINGS = [
    m_class1,
    m_class2,
    m_class3,
    m_class4,
    m_class5,
    m_class6,
    m_inst,
    m_grado,
]


class OrdinalColumnMapper(BaseEstimator, TransformerMixin):
    """Map column values to new values."""

    def __init__(
        self,
        columns: List[str] = COLUMNS_TO_MAP,
        mappings: List[Dict[str, int]] = MAPPINGS,
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
        *args,
        **kwargs,
    ):
        self.columns = columns
        self.args = args
        self.kwargs = kwargs
        self.encoder = OneHotEncoder(sparse_output=False, *args, **kwargs)

    def fit(self, X: pd.DataFrame, y=None):
        self.encoder.fit(X[self.columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = self.encoder.transform(X[self.columns])
        onehot_df = pd.DataFrame(
            data=data,
            columns=self.encoder.get_feature_names_out(),
        )
        X = X.drop(columns=self.columns)
        X = pd.concat([X, onehot_df], axis=1)
        return X


class NanInputer(BaseEstimator, TransformerMixin):
    """Inpute NA values."""

    def __init__(self, columns: List[str] | str = "auto"):
        self.columns = columns
