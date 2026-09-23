from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from config.config import LG_C, LG_MAXITER, DT_MAX_DEP, RF_EST, DT_MIN_SAMP_SPLIT,DT_MIN_SAMP_LEAF,RF_MAX_DEP, RF_MIN_SAMP
from src.features.pre_processing import preprocess


def model_pipeline_lg(df) -> Pipeline:
    lg = LogisticRegression(C= LG_C, max_iter= LG_MAXITER)
    pipeline = Pipeline([
        ("classifier", lg),
    ])
    return pipeline

def model_pipeline_rf(df) -> Pipeline:
    rf = RandomForestClassifier(n_estimators= RF_EST, max_depth=RF_MAX_DEP, min_samples_split= RF_MIN_SAMP)
    pipeline = Pipeline([
        ("classifier", rf),
    ])
    return pipeline

def model_pipeline_dt(df) -> Pipeline:
    dt = DecisionTreeClassifier(min_samples_leaf= DT_MIN_SAMP_LEAF, max_depth=DT_MAX_DEP, min_samples_split= DT_MIN_SAMP_SPLIT)
    pipeline = Pipeline([
        ("classifier", dt),
    ])
    return pipeline
