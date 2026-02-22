from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler


def make_logreg(preprocessor, max_iter: int = 3000):
    """Baseline multinomial com class_weight."""
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            multi_class="multinomial"
        ))
    ])


def make_logreg_with_undersampling(preprocessor, sampling_dict: dict, max_iter: int = 3000, random_state: int = 42):
    """Pipeline com RandomUnderSampler + LogisticRegression."""
    rus = RandomUnderSampler(sampling_strategy=sampling_dict, random_state=random_state)

    return ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("under", rus),
        ("clf", LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            multi_class="multinomial"
        ))
    ])