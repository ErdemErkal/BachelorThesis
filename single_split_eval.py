import subprocess
import sys
import os
import json
from copy import deepcopy
from types import SimpleNamespace


def run_command(command):
    subprocess.check_call(command, shell=True)


if not os.path.exists("MakeSurvivalCalibratedAgain"):
    run_command("git clone https://github.com/shi-ang/MakeSurvivalCalibratedAgain.git")

run_command(f'"{sys.executable}" -m pip install --upgrade pip')
run_command(f'"{sys.executable}" -m pip install -r MakeSurvivalCalibratedAgain/requirements.txt')
run_command(f'"{sys.executable}" -m pip install -U ucimlrepo')
run_command(f'"{sys.executable}" -m pip install SurvivalEVAL')

sys.path.append("MakeSurvivalCalibratedAgain")

os.environ["WANDB_MODE"] = "disabled"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_selector, ColumnTransformer

from utils.util_survival import survival_data_split, make_time_bins
from utils import set_seed

from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

from model import CoxPH, MTLR, CQRNN, LogNormalNN
from icp import ConformalSurvDist, CSDiPOT
from icp.scorer import QuantileRegressionNC, SurvivalPredictionNC

from pycox.models import DeepHitSingle, CoxTime
from pycox.models.cox_time import MLPVanillaCoxTime
import torchtuples as tt
from lifelines.fitters.weibull_aft_fitter import WeibullAFTFitter
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis

from SurvivalEVAL import SurvivalEvaluator, QuantileRegEvaluator

from sklearn.model_selection import train_test_split

data = pd.read_csv("results/datasets/veteran.csv")
data_test = pd.read_csv("results/create_distribution_shift/veteran_split_1_ess_1.0.tsv", sep="\t")
splits = json.loads(open("results/create_splits/veteran.json").read())

train_ix = np.array(splits["train"][1])
train_split = data.iloc[train_ix, :]

train_split = train_split.rename(columns={"status": "event"})

data_test = data_test.rename(columns={"status": "event"})

data_train, data_val, _ = survival_data_split(
    train_split,
    stratify_colname="both",
    frac_train=0.8,
    frac_val=0.2,
    frac_test=0.0,
    random_state=42,
)

num_cols = ["trt", "age", "karno", "diagtime", "prior"]
other_cols = ["celltype"]

X_train_num = data_train[num_cols].copy()
X_val_num = data_val[num_cols].copy()
X_test_num = data_test[num_cols].copy()

ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

X_train_cat = ohe.fit_transform(data_train[other_cols])
X_val_cat = ohe.transform(data_val[other_cols])
X_test_cat = ohe.transform(data_test[other_cols])

cat_feature_names = ohe.get_feature_names_out(other_cols)

X_train_cat = pd.DataFrame(X_train_cat, columns=cat_feature_names, index=data_train.index)
X_val_cat = pd.DataFrame(X_val_cat, columns=cat_feature_names, index=data_val.index)
X_test_cat = pd.DataFrame(X_test_cat, columns=cat_feature_names, index=data_test.index)

X_train = pd.concat([
    X_train_num.reset_index(drop=True),
    X_train_cat.reset_index(drop=True)
], axis=1)

X_val = pd.concat([
    X_val_num.reset_index(drop=True),
    X_val_cat.reset_index(drop=True)
], axis=1)

X_test = pd.concat([
    X_test_num.reset_index(drop=True),
    X_test_cat.reset_index(drop=True)
], axis=1)

X_train = X_train.astype(float)
X_val = X_val.astype(float)
X_test = X_test.astype(float)

scaler = StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

train_aft = X_train.copy()
train_aft["time"] = data_train["time"].values
train_aft["event"] = data_train["event"].astype(int).values

val_aft = X_val.copy()
val_aft["time"] = data_val["time"].values
val_aft["event"] = data_val["event"].astype(int).values

test_aft = X_test.copy()
test_aft["time"] = data_test["time"].values
test_aft["event"] = data_test["event"].astype(int).values

aft_base = WeibullAFTFitter(penalizer=0.01)
aft_base.fit(train_aft, duration_col="time", event_col="event")

t_train = train_aft["time"].values
e_train = train_aft["event"].values

t_val = val_aft["time"].values
e_val = val_aft["event"].values

t_test = test_aft["time"].values
e_test = test_aft["event"].values

t_train_ref = np.concatenate([t_train, t_val])
e_train_ref = np.concatenate([e_train, e_val])

x_test = test_aft.drop(columns=["time", "event"]).values


class PreFitQuantileRegressionNC(QuantileRegressionNC):
    def fit(self, train_set, val_set):
        return self


class PreFitSurvivalPredictionNC(SurvivalPredictionNC):
    def fit(self, train_set, val_set):
        return self


args = SimpleNamespace(
    model="AFT",
    n_quantiles=9,
    decensor_method="sampling",
    n_sample=1000,
    use_train=False,
    verbose=False,
    interpolate="Pchip",
    mono_method="ceil",
    seed=42,
)

surv_df = aft_base.predict_survival_function(test_aft)
time_coordinates = surv_df.index.values
surv_test = surv_df.values.T

time_coordinates = np.concatenate([np.array([0.0]), time_coordinates], axis=0)
surv_test = np.concatenate([np.ones((surv_test.shape[0], 1)), surv_test], axis=1)

evl_base = SurvivalEvaluator(
    surv_test,
    time_coordinates,
    t_test,
    e_test,
    t_train_ref,
    e_train_ref,
    predict_time_method="Median",
    interpolation="Pchip",
)

base_ci = evl_base.concordance(method="Harrell")[0]
base_dcal = evl_base.d_calibration()[0]

nc_csd = PreFitQuantileRegressionNC(deepcopy(aft_base), args)
icp_csd = ConformalSurvDist(
    nc_csd,
    condition=None,
    decensor_method=args.decensor_method,
    n_quantiles=args.n_quantiles,
)

icp_csd.fit(train_aft, val_aft)
icp_csd.calibrate(val_aft)

q_levels_csd, q_preds_csd = icp_csd.predict(x_test)

evl_csd = QuantileRegEvaluator(
    q_preds_csd,
    q_levels_csd,
    t_test,
    e_test,
    t_train_ref,
    e_train_ref,
    predict_time_method="Median",
    interpolation="Pchip",
)

csd_ci = evl_csd.concordance(method="Harrell")[0]
csd_dcal = evl_csd.d_calibration()[0]

nc_ipot = PreFitSurvivalPredictionNC(deepcopy(aft_base), args)
icp_ipot = CSDiPOT(
    nc_ipot,
    decensor_method=args.decensor_method,
    n_percentile=args.n_quantiles,
)

icp_ipot.fit(train_aft, val_aft)
icp_ipot.calibrate(val_aft)

q_levels_ipot, q_preds_ipot = icp_ipot.predict(x_test)

evl_ipot = QuantileRegEvaluator(
    q_preds_ipot,
    q_levels_ipot,
    t_test,
    e_test,
    t_train_ref,
    e_train_ref,
    predict_time_method="Median",
    interpolation="Pchip",
)

ipot_ci = evl_ipot.concordance(method="Harrell")[0]
ipot_dcal = evl_ipot.d_calibration()[0]

results = pd.DataFrame([
    {"method": "Baseline AFT", "Harrell_CI": base_ci, "D_CAL_pvalue": base_dcal},
    {"method": "AFT + CSD", "Harrell_CI": csd_ci, "D_CAL_pvalue": csd_dcal},
    {"method": "AFT + CSD-iPOT", "Harrell_CI": ipot_ci, "D_CAL_pvalue": ipot_dcal},
])

print(results)
