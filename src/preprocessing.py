import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def add_time_features(df: pd.DataFrame, date_col: str = "registry_date") -> pd.DataFrame:
    """Cria year, month, weekday a partir de date_col."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["weekday"] = out[date_col].dt.dayofweek
    return out


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),  # sparse por padrão
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def temporal_split(df: pd.DataFrame, date_col: str,
                   train_start: str, train_end: str,
                   val_start: str, val_end: str,
                   test_start: str, test_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")

    train = d[(d[date_col] >= train_start) & (d[date_col] <= train_end)]
    val   = d[(d[date_col] >= val_start) & (d[date_col] <= val_end)]
    test  = d[(d[date_col] >= test_start) & (d[date_col] <= test_end)]
    return train, val, test