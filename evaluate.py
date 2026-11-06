import os
import warnings

import click
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml, get_data_home
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model import NanoTabFM

# OpenML dataset IDs of the 30 binary classification tasks in TabArena.
TABARENA_BINARY = {
    46905: "Amazon_employee_access",
    46908: "APSFailure",
    46910: "bank-marketing",
    46911: "Bank_Customer_Churn",
    46912: "Bioresponse",
    46913: "blood-transfusion-service-center",
    46915: "churn",
    46916: "coil2000_insurance_policies",
    46918: "credit-g",
    46919: "credit_card_clients_default",
    46920: "customer_satisfaction_in_airline",
    46921: "diabetes",
    46922: "Diabetes130US",
    46924: "E-CommereShippingData",
    46927: "Fitness_Club",
    46929: "GiveMeSomeCredit",
    46930: "hazelnut-spread-contaminant-detection",
    46932: "heloc",
    46935: "HR_Analytics_Job_Change_of_Data_Scientists",
    46937: "in_vehicle_coupon_recommendation",
    46938: "Is-this-a-good-customer",
    46939: "kddcup09_appetency",
    46940: "Marketing_Campaign",
    46947: "online_shoppers_intention",
    46950: "polish_companies_bankruptcy",
    46952: "qsar-biodeg",
    46956: "seismic-bumps",
    46962: "taiwanese_bankruptcy_prediction",
    46969: "NATICUSdroid",
    46979: "jm1",
}


# Parsing the larger datasets takes minutes, so cache the subsampled tables.
@joblib.Memory(os.path.join(get_data_home(), "nano-tabfm"), verbose=0).cache
def load_dataset(data_id, max_rows=1000, max_features=10):
    """At most max_rows rows (stratified) and max_features random non-constant columns.

    Categorical columns are integer codes. Missing values are NaN.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = fetch_openml(data_id=data_id, as_frame=True, parser="liac-arff")
        data = dataset.data
        categorical = ~data.columns.isin(data.select_dtypes("number").columns)
        for column in data.columns[categorical]:
            data[column] = data[column].astype("category").cat.codes.replace(-1, np.nan)
        features = data.to_numpy(np.float32)
        labels = np.unique(dataset.target.astype(str), return_inverse=True)[1]
        if len(labels) > max_rows:
            features, _, labels, _ = train_test_split(
                features, labels, train_size=max_rows, stratify=labels, random_state=0
            )
        columns = np.flatnonzero(np.nanstd(features, axis=0) > 0)
    if len(columns) > max_features:
        columns = np.sort(np.random.default_rng(0).choice(columns, max_features, replace=False))
    return features[:, columns], labels, categorical[columns]


@click.command()
@click.argument("checkpoints", nargs=-1)
@click.option("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
def main(checkpoints, device):
    """Compare NanoTabFM weights (CHECKPOINTS) with baselines on TabArena's binary tasks."""
    torch.set_grad_enabled(False)

    models = {}
    for path in checkpoints:
        model = NanoTabFM().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        models[os.path.basename(path)] = model.eval()
    baselines = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(n_jobs=-1, random_state=0),
    }
    names = [*models, *baselines]
    width = max(10, *(len(name) + 2 for name in names))
    print(f"{'dataset (rows x features)':<48}" + "".join(f"{name:>{width}}" for name in names))

    scores = []
    for data_id, name in TABARENA_BINARY.items():
        features, labels, categorical = load_dataset(data_id)
        fold_scores = []
        folds = StratifiedKFold(5, shuffle=True, random_state=0)
        for train_rows, test_rows in folds.split(features, labels):
            # Fill missing values with the training mean, or the mode for categorical columns.
            train = pd.DataFrame(features[train_rows])
            fill = train.mean().where(~categorical, train.mode().iloc[0]).fillna(0)
            train_features = train.fillna(fill).to_numpy(np.float32)
            test_features = pd.DataFrame(features[test_rows]).fillna(fill).to_numpy(np.float32)
            train_labels, test_labels = labels[train_rows], labels[test_rows]

            probabilities = []
            for model in models.values():
                inputs = [
                    torch.from_numpy(x)[None].to(device)
                    for x in (train_features, train_labels, test_features)
                ]
                probabilities.append(model(*inputs).softmax(dim=-1)[0, :, 1].cpu().numpy())
            for estimator in baselines.values():
                estimator.fit(train_features, train_labels)
                probabilities.append(estimator.predict_proba(test_features)[:, 1])
            fold_scores.append([roc_auc_score(test_labels, p) for p in probabilities])

        scores.append(np.mean(fold_scores, axis=0))
        label = f"{name[:36]} ({features.shape[0]}x{features.shape[1]})"
        print(f"{label:<48}" + "".join(f"{score:{width}.4f}" for score in scores[-1]), flush=True)
    print(
        f"{'mean ROC AUC':<48}"
        + "".join(f"{score:{width}.4f}" for score in np.mean(scores, axis=0))
    )


if __name__ == "__main__":
    main()
