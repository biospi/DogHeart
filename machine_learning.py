from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import scipy.stats as stats
from sklearn.metrics import precision_score, recall_score, roc_auc_score, RocCurveDisplay, roc_curve

from utils import plot_heatmap_plotly

# Set matplotlib to use Times New Roman
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
from sklearn.metrics import RocCurveDisplay

from visu import show_db


def svm(out_dir, dataset_path, samples=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(dataset_path)
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
    df_X = df[features]
    df_y = df["label"]

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

    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X_pca, y, test_size=0.4, random_state=0
    # )

    rkf = RepeatedKFold(n_splits=2, n_repeats=100, random_state=0)
    aucs_train, aucs_test, precisions, sensitivities = [], [], [], []
    fprs_test, tprs_test = [], []
    fprs_train, tprs_train = [], []

    fig, ax = plt.subplots(figsize=(5, 5))
    X_pca = np.asarray(X_pca)
    y = np.asarray(y)

    for ifold, (train_index, test_index) in enumerate(rkf.split(X_pca, y)):
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the SVM model
        model = SVC(kernel="rbf", probability=True, random_state=0)
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

    # rkf = RepeatedKFold(
    #     n_splits=2,
    #     n_repeats=10,
    #     random_state=0,
    # )
    # aucs_train, aucs_test, precisions, sensitivities = [], [], [], []
    # fprs_test, tprs_test = [], []
    # fprs_train, tprs_train = [], []
    #
    # fig, ax = plt.subplots()
    # X_pca = np.asarray(X_pca)
    # y = np.asarray(y)
    # for ifold, (train_index, test_index) in enumerate(rkf.split(X_pca, y)):
    #     X_train, X_test = X_pca[train_index], X_pca[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #     # Train the SVM model
    #     # model = SVC(kernel='linear', probability=True, random_state=0, C=1e-7)
    #     model = SVC(kernel="rbf", probability=True, random_state=0)
    #     # model = LogisticRegression(random_state=0)
    #     model.fit(X_train, y_train)
    #     print(model)
    #
    #     y_pred = model.predict(X_test)
    #
    #     accuracy = accuracy_score(y_test, y_pred)
    #     print(f"Model Accuracy: {accuracy * 100:.2f}%")
    #     report = classification_report(y_test, y_pred, output_dict=True)
    #     print("Classification Report:\n")
    #     print(report)
    #
    #     # Calculate metrics for the current fold
    #     y_prob = model.predict_proba(X_test)[:, 1]
    #     auc_test = roc_auc_score(y_test, y_prob)
    #     auc_train = roc_auc_score(y_train, y_prob)
    #     precision = precision_score(y_test, y_pred)
    #     sensitivity = recall_score(y_test, y_pred)  # Sensitivity is recall
    #
    #     # Store metrics
    #     aucs_train.append(auc_train)
    #     aucs_test.append(auc_test)
    #     precisions.append(precision)
    #     sensitivities.append(sensitivity)
    #
    #     fpr, tpr, _ = roc_curve(y_test, y_prob)
    #     fprs_test.append(np.interp(np.linspace(0, 1, 100), fpr, tpr))  # Interpolate TPRs for mean ROC
    #     fprs_test[-1][0] = 0.0  # Fix the starting point at (0,0)
    #
    #     fpr, tpr, _ = roc_curve(y_train, y_prob)
    #     fprs_train.append(np.interp(np.linspace(0, 1, 100), fpr, tpr))  # Interpolate TPRs for mean ROC
    #     fprs_train[-1][0] = 0.0  # Fix the starting point at (0,0)
    #
    #     display_train = RocCurveDisplay.from_estimator(model, X_train, y_train, ax=ax)
    #     display_train.plot(color="orange", label="Train")
    #     display_test = RocCurveDisplay.from_estimator(
    #         model, X_test, y_test, ax=ax, plot_chance_level=True
    #     )
    #     display_test.plot(color="blue", label="Test")
    #
    #     class0_label = df_class_info[df_class_info["target"] == 0]["label"].tolist()
    #     class1_label = df_class_info[df_class_info["target"] == 1]["label"].tolist()
    #     class_0_precision = report["0"]["precision"]
    #     class_0_sensitivity = report["0"]["recall"]
    #     class_1_precision = report["1"]["precision"]
    #     class_1_sensitivity = report["1"]["recall"]
    #
    #     if ifold == 0:
    #         title = (
    #             f"{model} {dataset_path.stem}\n"
    #             f"Acuracy={accuracy:.02f}\n"
    #             f"{df["label"].value_counts().to_dict()}\n"
    #             f"Class 0 ({class0_label}) precision={class_0_precision:.02f} sensitivity{class_0_sensitivity:.02f}\n "
    #             f"Class 1 ({class1_label}) precision={class_1_precision:.02f} sensitivity{class_1_sensitivity:.02f}\n "
    #         )
    #         show_db(out_dir, title, model, X_pca, y, X_train, y_train, X_test, y_test, df_meta)
    #         # db = DBPlot(model, PCA(n_components=2))
    #         # db.fit(X_pca, y.values)
    #         # db.plot().show()
    #
    # # Calculate mean FPR and mean TPR
    # mean_fpr = np.linspace(0, 1, 100)
    # mean_tpr = np.mean(fprs_test, axis=0)
    # mean_tpr[-1] = 1.0  # Fix the ending point at (1,1)
    #
    # # Plot mean ROC curve in black
    # ax.plot(mean_fpr, mean_tpr, color='black', linestyle='--', label=f"Mean ROC (Test AUC: {np.mean(aucs_test):.2f})")
    #
    # ax.plot(mean_fpr, mean_tpr, color='red', linestyle='--', label=f"Mean ROC (Training AUC: {np.mean(aucs_train):.2f})")
    #
    # mean_auc_test, ci_auc_test = mean_ci(aucs_test)
    # mean_auc_train, ci_auc_train = mean_ci(aucs_train)
    # mean_precision, ci_precision = mean_ci(precisions)
    # mean_sensitivity, ci_sensitivity = mean_ci(sensitivities)
    #
    # print(f"Mean Test AUC: {mean_auc_test:.2f}, 95% CI: ({ci_auc_test[0]:.2f}, {ci_auc_test[1]:.2f})")
    # print(f"Mean Training AUC: {mean_auc_train:.2f}, 95% CI: ({ci_auc_train[0]:.2f}, {ci_auc_train[1]:.2f})")
    # print(f"Mean Precision: {mean_precision:.2f}, 95% CI: ({ci_precision[0]:.2f}, {ci_precision[1]:.2f})")
    # print(f"Mean Sensitivity: {mean_sensitivity:.2f}, 95% CI: ({ci_sensitivity[0]:.2f}, {ci_sensitivity[1]:.2f})")
    #
    # ax.set_title(f"SVM ROC curve\n{dataset_path.name}")
    # filepath = out_dir / "roc.png"
    # print(filepath)
    # fig.savefig(filepath)
    # fig.show()
    return df

def mean_ci(data):
    mean = np.mean(data)
    ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stats.sem(data))
    return mean, ci



if __name__ == "__main__":
    df_samples = svm(
        Path("output/ml"), Path("output/datasets5/cleaned_dataset_full.csv")
    )
