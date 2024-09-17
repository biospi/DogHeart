import random
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_curve, auc
from multiprocessing import Pool, Manager
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import precision_score

import matplotlib.pyplot as plt
np.random.seed(0)
from matplotlib import rcParams
# Set matplotlib to use Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']


def worker(
    out_dir,
    data,
    i,
    tot,
    bootstrap,
    xaxis_train,
    xaxis_test,
    auc_list_test,
    auc_list_train,
    tprs_test,
    tprs_train,
    fprs_test,
    fprs_train,
):
    all_test_y_list = []
    all_test_proba_list = []
    all_train_y_list = []
    all_train_proba_list = []
    prec_list_test = []
    prec_list_train = []
    print(f"bootstrap results progress {i}/{tot}...")

    all_test_proba = []
    all_test_y = []
    all_train_proba = []
    all_train_y = []
    for n, filepath in enumerate(bootstrap):
        loo_result = data[filepath.stem]
        y_pred_proba_test = np.array(loo_result["y_pred_proba_test"])
        if len(y_pred_proba_test.shape) > 1:
            y_pred_proba_test = y_pred_proba_test[:, 1]
        y_pred_proba_test = y_pred_proba_test.astype(np.float16)
        y_test = loo_result["y_test"]

        all_test_proba.extend(y_pred_proba_test)
        all_test_y.extend(y_test)
        all_test_y_list.extend(y_test)
        all_test_proba_list.extend(y_pred_proba_test)

        y_pred_proba_train = np.array(loo_result["y_pred_proba_train"])
        if len(y_pred_proba_train.shape) > 1:
            y_pred_proba_train = y_pred_proba_train[:, 1]
        y_pred_proba_train = y_pred_proba_train.astype(np.float16)
        y_train = loo_result["y_train"]
        all_train_proba.extend(y_pred_proba_train)
        all_train_y.extend(y_train)
        all_train_y_list.extend(y_train)
        all_train_proba_list.extend(y_pred_proba_train)

    fpr, tpr, thresholds = roc_curve(all_test_y, all_test_proba)
    tprs_test.append(tpr)
    fprs_test.append(fpr)
    roc_auc = auc(fpr, tpr)
    #print(roc_auc, all_test_y, all_test_proba)#todo check that all_test_y in bootstrap fold has 2 distinct values for roc
    auc_list_test.append(roc_auc)
    xaxis_test.append([fpr, tpr])
    # ax_roc_merge.plot(fpr, tpr, color="tab:blue", alpha=0.3, linewidth=1)

    fpr, tpr, thresholds = roc_curve(all_train_y, all_train_proba)
    tprs_train.append(tpr)
    fprs_train.append(fpr)
    roc_auc = auc(fpr, tpr)
    auc_list_train.append(roc_auc)
    xaxis_train.append([fpr, tpr])
    # ax_roc_merge.plot(fpr, tpr, color="tab:purple", alpha=0.3, linewidth=1)

    prec_list_test.append(
        precision_score(all_test_y, (np.array(all_test_proba) > 0.5).astype(int))
    )
    prec_list_train.append(
        precision_score(all_train_y, (np.array(all_train_proba) > 0.5).astype(int))
    )

    out = out_dir / str(i)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_test_y_list).to_pickle(out / "all_test_y_list.pkl")
    pd.DataFrame(all_test_proba_list).to_pickle(out / "all_test_proba_list.pkl")
    pd.DataFrame(all_train_y_list).to_pickle(out / "all_train_y_list.pkl")
    pd.DataFrame(all_train_proba_list).to_pickle(out / "all_train_proba_list.pkl")
    pd.DataFrame(prec_list_test).to_pickle(out / "prec_list_test.pkl")
    pd.DataFrame(prec_list_train).to_pickle(out / "prec_list_train.pkl")
    #print(f"{i}/{tot} done.")


def ninefive_confidence_interval(x):
    # boot_median = [np.median(np.random.choice(x, len(x))) for _ in range(iteration)]
    x.sort()
    lo_x_boot = np.percentile(x, 2.5)
    hi_x_boot = np.percentile(x, 97.5)
    # print(lo_x_boot, hi_x_boot)
    return lo_x_boot, hi_x_boot


def main(path=None, n_bootstrap=100, n_job=6):
    print("loading data...")
    paths = list(path.glob("*.json"))
    if len(paths) == 0:
        print("There are no .json files in the fold_data folder.")
        return
    out_dir = paths[0].parent.parent / "pickles"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    fig_roc_merge, ax_roc_merge = plt.subplots()
    for filepath in tqdm(paths):
        with open(filepath, "r") as fp:
            try:
                loo_result = json.load(fp)
            except Exception as e:
                print(e)
                return

            training_size = loo_result["training_shape"][0]
            testing_size = loo_result["testing_shape"][0]
            clf = f"{loo_result['clf']}({loo_result['clf_kernel']})"

            data[filepath.stem] = {
                "y_pred_proba_test": loo_result["y_pred_proba_test"],
                "y_test": loo_result["y_test"],
                "y_pred_proba_train": loo_result["y_pred_proba_train"],
                "y_train": loo_result["y_train"],
                "training_size": training_size,
                "testing_size": testing_size,
                "clf": clf,
            }

    print("start bootstrap...")
    pool = Pool(processes=n_job)
    with Manager() as manager:
        auc_list_test = manager.list()
        auc_list_train = manager.list()
        tprs_test = manager.list()
        tprs_train = manager.list()
        fprs_test = manager.list()
        fprs_train = manager.list()
        xaxis_train = manager.list()
        xaxis_test = manager.list()
        for i in range(n_bootstrap):
            bootstrap = np.random.choice(paths, size=len(paths), replace=True)
            print(bootstrap)
            pool.apply_async(
                worker,
                (
                    out_dir,
                    data,
                    i,
                    n_bootstrap,
                    bootstrap,
                    xaxis_train,
                    xaxis_test,
                    auc_list_test,
                    auc_list_train,
                    tprs_test,
                    tprs_train,
                    fprs_test,
                    fprs_train,
                ),
            )
        pool.close()
        pool.join()
        pool.terminate()
        #print("pool done.")
        xaxis_train = list(xaxis_train)
        xaxis_test = list(xaxis_test)
        auc_list_test = list(auc_list_test)
        auc_list_train = list(auc_list_train)

    all_test_y_list = []
    all_test_proba_list = []
    all_train_y_list = []
    all_train_proba_list = []
    prec_list_test = []
    prec_list_train = []
    for i in range(n_bootstrap):
        all_test_y_list.append(pd.read_pickle(out_dir / str(i) / "all_test_y_list.pkl").values.flatten())
        all_test_proba_list.append(pd.read_pickle(out_dir / str(i) / "all_test_proba_list.pkl").values.flatten())
        all_train_y_list.append(pd.read_pickle(out_dir / str(i) / "all_train_y_list.pkl").values.flatten())
        all_train_proba_list.append(pd.read_pickle(out_dir / str(i) / "all_train_proba_list.pkl").values.flatten())
        prec_list_test.append(pd.read_pickle(out_dir / str(i) / "prec_list_test.pkl").values.flatten())
        prec_list_train.append(pd.read_pickle(out_dir / str(i) / "prec_list_train.pkl").values.flatten())

    # prec_list_test = np.mean(prec_list_test)
    # prec_list_train = np.mean(prec_list_train)

    print("building roc...")
    median_auc_test = np.nanmedian(auc_list_test)
    lo_test_auc, hi_test_auc = ninefive_confidence_interval(auc_list_test)
    print(
        f"Testing AUC = {median_auc_test:.2f}({lo_test_auc:.1f}, {hi_test_auc:.1f})"
    )

    median_auc_train = np.nanmedian(auc_list_train)
    lo_train_auc, hi_train_auc = ninefive_confidence_interval(auc_list_train)
    print(
        f"Training AUC = {median_auc_train:.2f}({lo_train_auc:.1f}, {hi_train_auc:.1f})"
    )

    median_prec_test = np.nanmedian(prec_list_test)
    lo_test_prec, hi_test_prec = ninefive_confidence_interval(prec_list_test)
    print(
        f"Testing prec = {median_prec_test:.2f}({lo_test_prec:.1f}, {hi_test_prec:.1f})"
    )

    median_prec_train = np.nanmedian(prec_list_train)
    lo_train_prec, hi_train_prec = ninefive_confidence_interval(prec_list_train)
    print(
        f"Training prec = {median_prec_train:.2f}({lo_train_prec:.1f}, {hi_train_prec:.1f})"
    )

    try:
        xaxis_train_ = random.sample(xaxis_train, 10)
    except ValueError as e:
        print(e)
        xaxis_train_ = xaxis_train

    for fpr, tpr in xaxis_train_:
        ax_roc_merge.plot(fpr, tpr, color="tab:purple", alpha=0.3, linewidth=1)

    xaxis_test_ = random.sample(xaxis_test, 10)
    for fpr, tpr in xaxis_test_:
        ax_roc_merge.plot(fpr, tpr, color="tab:blue", alpha=0.3, linewidth=1)

    ax_roc_merge.plot(
        [0, 1], [0, 1], linestyle="--", lw=2, color="orange", label="Chance", alpha=1
    )

    label = f"Testing (Median AUC = {median_auc_test:.2f}({lo_test_auc:.1f}, {hi_test_auc:.1f})"

    mean_fpr_test, mean_tpr_test = [], []
    for y_list, proba_list in zip(all_test_y_list, all_test_proba_list):
        mean_fpr, mean_tpr, thresholds = roc_curve(
            y_list, proba_list
        )
        mean_fpr_test.append(mean_fpr)
        mean_tpr_test.append(mean_tpr)

    max_length = max(len(arr) for arr in mean_fpr_test)
    mean_fpr_test = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=1) for arr in mean_fpr_test]
    mean_tpr_test = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=1) for arr in mean_tpr_test]

    mean_fpr_test = np.median(mean_fpr_test, axis=0)
    mean_tpr_test = np.median(mean_tpr_test, axis=0)

    ax_roc_merge.plot(
        mean_fpr_test, mean_tpr_test, color="black", label=label, lw=2, alpha=1
    )
    ax_roc_merge.tick_params(axis='x', labelsize=18)  # Adjust the fontsize as needed for the x-axis
    ax_roc_merge.tick_params(axis='y', labelsize=18)
    ax_roc_merge.set_xlabel("False positive rate", fontsize=22)
    ax_roc_merge.set_ylabel("True positive rate", fontsize=22)
    ax_roc_merge.legend(loc="lower right", fontsize=14)
    # fig.show()

    label = f"Training (Median AUC = {median_auc_train:.2f}({lo_train_auc:.1f}, {hi_train_auc:.1f})"
    mean_fpr_train, mean_tpr_train = [], []
    for y_list, proba_list in zip(all_train_y_list, all_train_proba_list):
        mean_fpr, mean_tpr, thresholds = roc_curve(
            y_list, proba_list
        )
        mean_fpr_train.append(mean_fpr)
        mean_tpr_train.append(mean_tpr)

    max_length = max(len(arr) for arr in mean_fpr_train)
    mean_fpr_train = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=1) for arr in mean_fpr_train]
    mean_tpr_train = [np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=1) for arr in mean_tpr_train]

    mean_fpr_train = np.median(mean_fpr_train, axis=0)
    mean_tpr_train = np.median(mean_tpr_train, axis=0)

    ax_roc_merge.plot(
        mean_fpr_train, mean_tpr_train, color="red", label=label, lw=2, alpha=1
    )
    ax_roc_merge.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        #title=f"Receiver operating characteristic (n_bootstrap={n_bootstrap})",
    )
    ax_roc_merge.legend(loc="lower right")

    fig_roc_merge.tight_layout()
    path_ = path / "roc_curve"
    path_.mkdir(parents=True, exist_ok=True)
    # final_path = path / f"{tag}_roc_{classifier_name}.png"
    # print(final_path)
    # fig.savefig(final_path)

    final_path = (
        path_
        / f"{n_bootstrap}.png"
    )
    print(final_path)

    fig_roc_merge.set_size_inches(6, 6)
    fig_roc_merge.tight_layout()
    fig_roc_merge.savefig(final_path, dpi=500)

    return [
        f"{median_auc_test:.2f} ({lo_test_auc:.2f}-{hi_test_auc:.2f})",
        f"{median_auc_train:.2f} ({lo_train_auc:.2f}-{hi_train_auc:.2f})",
        f"{median_prec_test:.2f} ({lo_test_prec:.2f}-{hi_test_prec:.2f})",
        f"{median_prec_train:.2f} ({lo_train_prec:.2f}-{hi_train_prec:.2f})",
        training_size,
        testing_size,
        clf,
        median_auc_test,
        median_auc_train,
        auc_list_test,
        auc_list_train,
        paths
    ]
