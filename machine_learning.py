from pathlib import Path
import json

from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import scipy.stats as stats
from sklearn.metrics import precision_score, recall_score, roc_auc_score, RocCurveDisplay, roc_curve
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    balanced_accuracy_score,
    auc,
    roc_curve,
)
from sklearn.metrics import precision_recall_fscore_support

import boot_roc_curve
from utils import plot_heatmap_plotly, get_2d_histogram

# Set matplotlib to use Times New Roman
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
from sklearn.metrics import RocCurveDisplay

from visu import show_db


def svm_loocv(out_dir, dataset_path, participant_to_keep=None, n_bootstrap=10, n_job=6):
    df, df_X, df_y, features = get_samples(out_dir, dataset_path)

    # Target variable binary encoding
    y = df_y.apply(lambda x: 1 if x in ["B2"] else 0)
    df["target"] = y

    df = df.reset_index()

    # Impute missing values with the median
    X_imputed = df_X.fillna(df_X.median())

    # Standardize the features
    X_scaled = StandardScaler().fit_transform(X_imputed)

    # Apply PCA (optional, adjust n_components if needed)
    pca = PCA(n_components=20)
    X_pca = pca.fit_transform(X_scaled)

    # Convert the data to arrays for model input
    X_pca = np.asarray(X_pca)
    y = np.asarray(y)

    # Leave-One-Group-Out cross-validation by participant_id
    participant_ids = df['participant_id']  # Make sure you have a column with participant IDs
    logo = LeaveOneGroupOut()

    # Perform LOOCV
    all_pred = []
    all_y = []
    for ifold, (train_idx, test_idx) in enumerate(logo.split(X_pca, y, groups=participant_ids)):
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf_kernel = "rbf"
        # Train the SVM model
        clf = SVC(kernel=clf_kernel, probability=True, random_state=0)
        clf.fit(X_train, y_train)

        # Test the model
        y_pred_test = clf.predict(X_test)
        if hasattr(clf, "decision_function"):
            y_pred_proba_test_df = clf.decision_function(X_test)
            y_pred_proba_test = clf.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba_test = clf.predict_proba(X_test)[:, 1]
            y_pred_proba_test_df = y_pred_proba_test
        accuracy = balanced_accuracy_score(y_test, y_pred_test)
        precision, recall, fscore, support = precision_recall_fscore_support(
            y_test, y_pred_test
        )
        precision = np.append(precision, precision[0])
        recall = np.append(recall, recall[0])
        fscore = np.append(fscore, fscore[0])
        support = np.append(support, support[0])

        y_pred_train = clf.predict(X_train)
        if hasattr(clf, "decision_function"):
            y_pred_proba_train_df = clf.decision_function(X_train)
            y_pred_proba_train = clf.predict_proba(X_train)[:, 1]
        else:
            y_pred_proba_train = clf.predict_proba(X_train)[:, 1]
            y_pred_proba_train_df = y_pred_proba_train

        accuracy_train = balanced_accuracy_score(y_train, y_pred_train)
        (
            precision_train,
            recall_train,
            fscore_train,
            support_train,
        ) = precision_recall_fscore_support(y_train, y_pred_train)

        fold_result = {
            "clf": type(clf).__name__,
            "clf_kernel": clf_kernel,
            "training_shape": X_train.shape,
            "testing_shape": X_test.shape,
            "target": int(y_test.tolist()[0]),
            "accuracy": float(accuracy),
            "accuracy_train": float(accuracy_train),
            "y_test": y_test.tolist(),
            "y_train": y_train.tolist(),
            "y_pred_test": y_pred_test.tolist(),
            "y_pred_train": y_pred_train.tolist(),
            "y_pred_proba_test": y_pred_proba_test.tolist(),
            "y_pred_proba_train": y_pred_proba_train.tolist(),
            "y_pred_proba_test_df": y_pred_proba_test_df.tolist(),
            "y_pred_proba_train_df": y_pred_proba_train_df.tolist(),
            "ids_test": participant_ids[test_idx].tolist(),
            "ids_train": participant_ids[train_idx].tolist(),
            "test_precision_score_0": float(precision[0]),
            "test_precision_score_1": float(precision[1]),
            "test_recall_0": float(recall[0]),
            "test_recall_1": float(recall[1]),
            "test_fscore_0": float(fscore[0]),
            "test_fscore_1": float(fscore[1]),
            "test_support_0": float(support[0]),
            "test_support_1": float(support[1]),
            "train_precision_score_0": float(precision_train[0]),
            "train_precision_score_1": float(precision_train[1]),
            "train_recall_0": float(recall_train[0]),
            "train_recall_1": float(recall_train[1]),
            "train_fscore_0": float(fscore_train[0]),
            "train_fscore_1": float(fscore_train[1]),
            "train_support_0": float(support_train[0]),
            "train_support_1": float(support_train[1])
        }
        all_pred.extend(y_pred_proba_test.tolist())
        all_y.extend(y_test.tolist())

        print("export result to json...")
        out_ml_dir = out_dir / "fold_data"
        out_ml_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_ml_dir / f"{str(ifold).zfill(4)}_result.json"
        print(filepath)
        with open(str(filepath), "w") as fp:
            json.dump(fold_result, fp)

    fpr, tpr, thresholds = roc_curve(all_y, all_pred)
    auc = roc_auc_score(all_y, all_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig('roc_curve.png')
    plt.show()

    res = boot_roc_curve.main(
        out_ml_dir, n_bootstrap=n_bootstrap, n_job=n_job
    )

    return df


def ml_l2out(model, out_dir, dataset_path, participant_to_keep=None, samples=None):
    df, df_X, df_y, features = get_samples(out_dir, dataset_path)

    df_meta = (
        df["clinic"].astype(str) + df["home"].astype(str) + df["label"].astype(str)
    )
    df_meta = (
        df_meta.str.replace("nan", "")
        .str.replace("Home", "")
        .str.replace("Walk", "W")
        .str.replace("Food", "F")
        .str.replace("House", "H")
        .str.replace("Sleep", "S")
        .str.replace("Garden", "G")
        .str.replace("garden", "G")
        .str.replace("Dinner", "D")
        .str.replace("Rest", "R")
        .str.replace("B", "")
    )

    y = df_y.apply(lambda x: 1 if x in ["B2"] else 0)
    df["target"] = y
    df_class_info = df[["target", "label"]].drop_duplicates().sort_values("target")
    print("Class info:")
    print(df_class_info)

    # imputation
    X_imputed = df_X.fillna(df_X.median())
    # df_X_imputed = df_X.fillna(df_X.median())
    # df_X_imputed = df_X.fillna(0)

    print("Class numbers:")
    print(df["label"].value_counts())

    z = df_X.values
    x = np.asarray(features)
    sample_labels = [f"{i} {x}" for i, x in enumerate(df["label"])]
    plot_heatmap_plotly(
        z,
        x,
        sample_labels,
        out_dir,
        title="samples",
        filename=f"samples_{dataset_path.stem}.html",
    )

    # X_imputed = Normalizer(norm="l1").transform(X_imputed)
    # X_imputed = np.log(anscombe(X_imputed))

    X_scaled = StandardScaler().fit_transform(X_imputed)

    pca = PCA(n_components=20)
    X_pca = pca.fit_transform(X_scaled)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X_pca, y, test_size=0.4, random_state=0
    # )

    #rkf = RepeatedKFold(n_splits=2, n_repeats=100, random_state=0)
    aucs_train, aucs_test, precisions, sensitivities = [], [], [], []
    fprs_test, tprs_test = [], []
    fprs_train, tprs_train = [], []

    fig, ax = plt.subplots(figsize=(5, 5))
    X_pca = np.asarray(X_pca)
    y = np.asarray(y)

    participant_ids = df['participant_id'].unique()  # Ensure participant_ids are unique

    # Get all combinations of two participants
    group_combinations = list(combinations(participant_ids, 2))

    # Perform LTGO
    for ifold, (group1, group2) in enumerate(group_combinations):
        # Get indices for these two groups
        test_idx = np.where((df['participant_id'] == group1) | (df['participant_id'] == group2))[0]
        train_idx = np.where(~((df['participant_id'] == group1) | (df['participant_id'] == group2)))[0]

        # Split data based on the indices
        X_train, X_test = X_pca[train_idx], X_pca[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_test)) == 1:
            continue
        # Now you can proceed with your model training and evaluation in each fold
        print(f"Fold {ifold + 1}: Leaving out participants {group1} and {group2}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        report = classification_report(y_test, y_pred, output_dict=True)
        print("Classification Report:\n", report)

        # Calculate metrics for the current fold
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]

        auc_test = roc_auc_score(y_test, y_prob_test)
        if auc_test < 0.5:
            auc_test = 0.5
        auc_train = roc_auc_score(y_train, y_prob_train)
        if auc_train < 0.5:
            auc_train = 0.5
        precision = precision_score(y_test, y_pred)
        sensitivity = recall_score(y_test, y_pred)  # Sensitivity is recall

        # Store metrics
        aucs_train.append(auc_train)
        aucs_test.append(auc_test)
        precisions.append(precision)
        sensitivities.append(sensitivity)

        # Compute ROC curves for both train and test sets
        fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
        fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)

        # Interpolate the ROC curves to have same FPR values for averaging
        interp_fpr = np.linspace(0, 1, 100)
        tprs_test.append(np.interp(interp_fpr, fpr_test, tpr_test))
        tprs_test[-1][0] = 0.0  # Fix the starting point at (0,0)

        tprs_train.append(np.interp(interp_fpr, fpr_train, tpr_train))
        tprs_train[-1][0] = 0.0  # Fix the starting point at (0,0)

        # Plot ROC curve for each fold
        if auc_test > 0.5:
            ax.plot(fpr_test, tpr_test, color="tab:blue", alpha=0.3)
        if auc_train > 0.5:
            ax.plot(fpr_train, tpr_train, color="tab:orange", alpha=0.3)

        if ifold == 0:
            class0_label = df_class_info[df_class_info["target"] == 0]["label"].tolist()
            class1_label = df_class_info[df_class_info["target"] == 1]["label"].tolist()
            class_0_precision = report["0"]["precision"]
            class_0_sensitivity = report["0"]["recall"]
            class_1_precision = report["1"]["precision"]
            class_1_sensitivity = report["1"]["recall"]
            title = (
                f"{model} {dataset_path.stem}\n"
                f"Acuracy={accuracy:.02f}\n"
                f"{df["label"].value_counts().to_dict()}\n"
                f"Class 0 ({class0_label}) precision={class_0_precision:.02f} sensitivity{class_0_sensitivity:.02f}\n "
                f"Class 1 ({class1_label}) precision={class_1_precision:.02f} sensitivity{class_1_sensitivity:.02f}\n "
            )
            #show_db(out_dir, title, model, X_pca, y, X_train, y_train, X_test, y_test, df_meta)
            # db = DBPlot(model, PCA(n_components=2))
            # db.fit(X_pca, y.values)
            # db.plot().show()


    # Calculate mean FPR and TPR for test set
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr_test = np.mean(tprs_test, axis=0)
    mean_tpr_test[-1] = 1.0  # Fix the ending point at (1,1)

    # Calculate mean FPR and TPR for train set
    mean_tpr_train = np.mean(tprs_train, axis=0)
    mean_tpr_train[-1] = 1.0

    # Confidence intervals and metrics
    mean_auc_test, ci_auc_test = mean_ci(aucs_test)
    mean_auc_train, ci_auc_train = mean_ci(aucs_train)
    mean_precision, ci_precision = mean_ci(precisions)
    mean_sensitivity, ci_sensitivity = mean_ci(sensitivities)

    # Plot mean ROC curve in black for test and red for train
    ax.plot(mean_fpr, mean_tpr_test, color='black', linestyle='--',
            label=f"Mean ROC (Test AUC: {mean_auc_test:.2f}, 95% CI: ({ci_auc_test[0]:.2f}, {ci_auc_test[1]:.2f})")
    ax.plot(mean_fpr, mean_tpr_train, color='red', linestyle='--',
            label=f"Mean ROC (Train AUC: {mean_auc_train:.2f}, 95% CI: ({ci_auc_train[0]:.2f}, {ci_auc_train[1]:.2f})")

    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance level', alpha=0.8)

    # Print the mean and CI
    test_auc_label = f"Mean Test AUC: {mean_auc_test:.2f}, 95% CI: ({ci_auc_test[0]:.2f}, {ci_auc_test[1]:.2f})"
    print(test_auc_label)
    train_auc_label = f"Mean Training AUC: {mean_auc_train:.2f}, 95% CI: ({ci_auc_train[0]:.2f}, {ci_auc_train[1]:.2f})"
    print(train_auc_label)
    precision_label = f"Mean Precision: {mean_precision:.2f}, 95% CI: ({ci_precision[0]:.2f}, {ci_precision[1]:.2f})"
    print(precision_label)
    sensitivity_label = f"Mean Sensitivity: {mean_sensitivity:.2f}, 95% CI: ({ci_sensitivity[0]:.2f}, {ci_sensitivity[1]:.2f})"
    print(sensitivity_label)

    # Finalize plot
    ax.set_title(f"SVM ROC curve\n"
                 f"Leave2Out folds:{ifold}\n"
                 f"{dataset_path.name}\n"
                 f"{precision_label}\n"
                 f"{sensitivity_label}")
    ax.legend(loc='lower right')
    fig.tight_layout()
    filepath = out_dir / "roc_l2out.png"
    fig.savefig(filepath)
    plt.show()
    return df


def get_samples(out_dir, dataset_path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(dataset_path)
    if participant_to_keep is not None:
        df_raw = df_raw[df_raw["participant_id"].isin(participant_to_keep)]

    features_columns_pulse = [x for x in df_raw.columns if "pulse" in x.lower()]
    features_columns_counts = [x for x in df_raw.columns if "count" in x.lower()]
    features_columns_mag = [x for x in df_raw.columns if "magnitude" in x.lower()]
    features_columns_acc = [x for x in df_raw.columns if "acc" in x.lower()]
    df_raw["unique_values_pulse"] = df_raw[features_columns_pulse].apply(
        lambda row: row.nunique(), axis=1
    )

    df_raw = df_raw[df_raw["unique_values_pulse"] > 2000].copy()
    df = df_raw
    df = df[df["mean"] > 3]
    df = df[
        df["missingness_percentage"] <= 40
    ]  # keep samples with less than 50% of missing points
    # df = df[df["pos_value_count"] > 100] #keep samples with positive values  above threshold
    df = df[df["label"] != "C"]  # remove C category
    #df = df[pd.isna(df["clinic"])]  # only keep home data

    features = features_columns_pulse

    df_pulse = df[features_columns_pulse]
    df_pulse = df_pulse.fillna(df_pulse.median())
    df_count = df[features_columns_counts]
    df_count = df_count.fillna(df_count.median())

    df_X = df[features]
    df_y = df["label"]

    ###################################
    # Here we build a new dataset by creating 2d histogram with the pulse and activity count data
    hist_samples = []
    for (index, row_pulse), (index, row_count) in zip(df_pulse.iterrows(), df_count.iterrows()):
        data1 = row_pulse.tolist()
        data2 = row_count.tolist()[0:len(data1)]
        hist = get_2d_histogram(data1, data2, plot=False)
        hist_sample = hist.flatten().tolist()
        hist_samples.append(hist_sample)

    df_hist = pd.DataFrame(hist_samples)
    df_X = df_hist
    ###################################
    return df, df_X, df_y, features


def ml_kfold(model, out_dir, dataset_path, participant_to_keep=None, samples=None):
    df, df_X, df_y, features = get_samples(out_dir, dataset_path)

    df_meta = (
        df["clinic"].astype(str) + df["home"].astype(str) + df["label"].astype(str)
    )
    df_meta = (
        df_meta.str.replace("nan", "")
        .str.replace("Home", "")
        .str.replace("Walk", "W")
        .str.replace("Food", "F")
        .str.replace("House", "H")
        .str.replace("Sleep", "S")
        .str.replace("Garden", "G")
        .str.replace("garden", "G")
        .str.replace("Dinner", "D")
        .str.replace("Rest", "R")
        .str.replace("B", "")
    )

    y = df_y.apply(lambda x: 1 if x in ["B2"] else 0)
    df["target"] = y
    df_class_info = df[["target", "label"]].drop_duplicates().sort_values("target")
    print("Class info:")
    print(df_class_info)

    # imputation
    X_imputed = df_X.fillna(df_X.median())
    # df_X_imputed = df_X.fillna(df_X.median())
    # df_X_imputed = df_X.fillna(0)

    print("Class numbers:")
    print(df["label"].value_counts())

    z = df_X.values
    x = np.asarray(features)
    sample_labels = [f"{i} {x}" for i, x in enumerate(df["label"])]
    plot_heatmap_plotly(
        z,
        x,
        sample_labels,
        out_dir,
        title="samples",
        filename=f"samples_{dataset_path.stem}.html",
    )

    # X_imputed = Normalizer(norm="l1").transform(X_imputed)
    # X_imputed = np.log(anscombe(X_imputed))

    X_scaled = StandardScaler().fit_transform(X_imputed)

    pca = PCA(n_components=15)
    X_pca = pca.fit_transform(X_scaled)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X_pca, y, test_size=0.4, random_state=0
    # )

    rkf = RepeatedKFold(n_splits=5, n_repeats=100, random_state=0)
    aucs_train, aucs_test, precisions, sensitivities = [], [], [], []
    fprs_test, tprs_test = [], []
    fprs_train, tprs_train = [], []

    fig, ax = plt.subplots(figsize=(5, 5))
    X_pca = np.asarray(X_pca)
    y = np.asarray(y)

    for ifold, (train_index, test_index) in enumerate(rkf.split(X_pca, y)):
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        report = classification_report(y_test, y_pred, output_dict=True)
        print("Classification Report:\n", report)

        # Calculate metrics for the current fold
        y_prob_train = model.predict_proba(X_train)[:, 1]
        y_prob_test = model.predict_proba(X_test)[:, 1]

        auc_test = roc_auc_score(y_test, y_prob_test)
        if auc_test < 0.5:
            auc_test = 0.5
        auc_train = roc_auc_score(y_train, y_prob_train)
        if auc_train < 0.5:
            auc_train = 0.5
        precision = precision_score(y_test, y_pred)
        sensitivity = recall_score(y_test, y_pred)  # Sensitivity is recall

        # Store metrics
        aucs_train.append(auc_train)
        aucs_test.append(auc_test)
        precisions.append(precision)
        sensitivities.append(sensitivity)

        # Compute ROC curves for both train and test sets
        fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
        fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)

        # Interpolate the ROC curves to have same FPR values for averaging
        interp_fpr = np.linspace(0, 1, 100)
        tprs_test.append(np.interp(interp_fpr, fpr_test, tpr_test))
        tprs_test[-1][0] = 0.0  # Fix the starting point at (0,0)

        tprs_train.append(np.interp(interp_fpr, fpr_train, tpr_train))
        tprs_train[-1][0] = 0.0  # Fix the starting point at (0,0)

        # Plot ROC curve for each fold
        if auc_test > 0.5:
            ax.plot(fpr_test, tpr_test, color="tab:blue", alpha=0.3)
        if auc_train > 0.5:
            ax.plot(fpr_train, tpr_train, color="tab:orange", alpha=0.3)

        if ifold == 0:
            class0_label = df_class_info[df_class_info["target"] == 0]["label"].tolist()
            class1_label = df_class_info[df_class_info["target"] == 1]["label"].tolist()
            class_0_precision = report["0"]["precision"]
            class_0_sensitivity = report["0"]["recall"]
            class_1_precision = report["1"]["precision"]
            class_1_sensitivity = report["1"]["recall"]
            title = (
                f"{model} {dataset_path.stem}\n"
                f"Acuracy={accuracy:.02f}\n"
                f"{df["label"].value_counts().to_dict()}\n"
                f"Class 0 ({class0_label}) precision={class_0_precision:.02f} sensitivity{class_0_sensitivity:.02f}\n "
                f"Class 1 ({class1_label}) precision={class_1_precision:.02f} sensitivity{class_1_sensitivity:.02f}\n "
            )
            #show_db(out_dir, title, model, X_pca, y, X_train, y_train, X_test, y_test, df_meta)
            # db = DBPlot(model, PCA(n_components=2))
            # db.fit(X_pca, y.values)
            # db.plot().show()


    # Calculate mean FPR and TPR for test set
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr_test = np.mean(tprs_test, axis=0)
    mean_tpr_test[-1] = 1.0  # Fix the ending point at (1,1)

    # Calculate mean FPR and TPR for train set
    mean_tpr_train = np.mean(tprs_train, axis=0)
    mean_tpr_train[-1] = 1.0

    # Confidence intervals and metrics
    mean_auc_test, ci_auc_test = mean_ci(aucs_test)
    mean_auc_train, ci_auc_train = mean_ci(aucs_train)
    mean_precision, ci_precision = mean_ci(precisions)
    mean_sensitivity, ci_sensitivity = mean_ci(sensitivities)

    # Plot mean ROC curve in black for test and red for train
    ax.plot(mean_fpr, mean_tpr_test, color='black', linestyle='--',
            label=f"Mean ROC (Test AUC: {mean_auc_test:.2f}, 95% CI: ({ci_auc_test[0]:.2f}, {ci_auc_test[1]:.2f})")
    ax.plot(mean_fpr, mean_tpr_train, color='red', linestyle='--',
            label=f"Mean ROC (Train AUC: {mean_auc_train:.2f}, 95% CI: ({ci_auc_train[0]:.2f}, {ci_auc_train[1]:.2f})")

    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance level', alpha=0.8)

    # Print the mean and CI
    test_auc_label = f"Mean Test AUC: {mean_auc_test:.2f}, 95% CI: ({ci_auc_test[0]:.2f}, {ci_auc_test[1]:.2f})"
    print(test_auc_label)
    train_auc_label = f"Mean Training AUC: {mean_auc_train:.2f}, 95% CI: ({ci_auc_train[0]:.2f}, {ci_auc_train[1]:.2f})"
    print(train_auc_label)
    precision_label = f"Mean Precision: {mean_precision:.2f}, 95% CI: ({ci_precision[0]:.2f}, {ci_precision[1]:.2f})"
    print(precision_label)
    sensitivity_label = f"Mean Sensitivity: {mean_sensitivity:.2f}, 95% CI: ({ci_sensitivity[0]:.2f}, {ci_sensitivity[1]:.2f})"
    print(sensitivity_label)

    # Finalize plot
    ax.set_title(f"SVM ROC curve\n"
                 f"{rkf}\n"
                 f"{dataset_path.name}\n"
                 f"{precision_label}\n"
                 f"{sensitivity_label}")
    ax.legend(loc='lower right')
    fig.tight_layout()
    filepath = out_dir / "roc.png"
    fig.savefig(filepath)
    plt.show()
    return df

def mean_ci(data):
    mean = np.mean(data)
    ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stats.sem(data))
    return mean, ci


if __name__ == "__main__":
    participant_to_keep = [4, 7, 10, 11, 14, 15, 17, 20, 21, 22, 23, 25, 26, 28, 29]

    ml_kfold(
        SVC(kernel="rbf", probability=True, random_state=0),
        Path("output/ml"), Path("output/datasets5/cleaned_dataset_full.csv"),
        participant_to_keep=participant_to_keep
    )

    ml_kfold(
        KNeighborsClassifier(n_neighbors=3),
        Path("output/ml"), Path("output/datasets5/cleaned_dataset_full.csv"),
        participant_to_keep=participant_to_keep
    )

    ml_kfold(
        LogisticRegression(),
        Path("output/ml"), Path("output/datasets5/cleaned_dataset_full.csv"),
        participant_to_keep=participant_to_keep
    )

    ml_l2out(
        SVC(kernel="rbf", probability=True, random_state=0),
        Path("output/ml"), Path("output/datasets5/cleaned_dataset_full.csv"),
        participant_to_keep=participant_to_keep
    )

    ml_kfold(
        KNeighborsClassifier(n_neighbors=3),
        Path("output/ml"), Path("output/datasets5/cleaned_dataset_full.csv"),
        participant_to_keep=participant_to_keep,
    )

    ml_l2out(
        LogisticRegression(),
        Path("output/ml"), Path("output/datasets5/cleaned_dataset_full.csv"),
        participant_to_keep=participant_to_keep,
    )
