import numpy as np
import pandas as pd
from pathlib import Path

from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams

from utils import plot_heatmap_plotly

# Set matplotlib to use Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
from sklearn.metrics import RocCurveDisplay

from visu import show_db


def svm(out_dir, dataset_path, samples=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(dataset_path)
    features_columns = [x for x in df_raw.columns if x[0] == 'x']
    df_raw['unique_values'] = df_raw.apply(lambda row: row.nunique(), axis=1)

    df = df_raw[df_raw['unique_values'] > 2000]
    #df = df_raw

    df_X = df[features_columns]
    df_y = df['label']

    y = df_y.apply(lambda x: 1 if x in ['B2', 'C'] else 0)
    df["target"] = y
    df_class_info = df[["target", "label"]].drop_duplicates().sort_values('target')
    print("Class info:")
    print(df_class_info)

    #imputation
    df_X_imputed = df_X.fillna(df_X.mean())
    #df_X_imputed = df_X.fillna(df_X.median())
    #df_X_imputed = df_X.fillna(0)

    print("Class numbers:")
    print(df['label'].value_counts())

    z = df_X.values
    x = np.asarray(features_columns)
    sample_labels = [f"{i} {x}" for i, x in enumerate(df['label'])]
    plot_heatmap_plotly(z, x, sample_labels, out_dir, title="samples", filename=f"samples_{dataset_path.stem}.html")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_X_imputed)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.4, random_state=0)

    # Train the SVM model
    #model = SVC(kernel='linear', probability=True, random_state=0, C=1e-7)
    model = SVC(kernel='rbf', probability=True, random_state=0)
    #model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    print(model)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Classification Report:\n")
    print(report)

    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_train, y_train, ax=ax, name='Training')
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, plot_chance_level=True, name='Testing')
    ax.set_title(f'SVM ROC curve\n{dataset_path.name}')
    filepath = out_dir / "roc.png"
    print(filepath)
    fig.savefig(filepath)
    fig.show()

    class0_label = df_class_info[df_class_info['target'] == 0]['label'].tolist()
    class1_label = df_class_info[df_class_info['target'] == 1]['label'].tolist()
    class_0_precision = report['0']['precision']
    class_0_sensitivity = report['0']['recall']
    class_1_precision = report['1']['precision']
    class_1_sensitivity = report['1']['recall']
    title = (f"{model} {dataset_path.stem}\n"
             f"Acuracy={accuracy:.02f}\n"
             f"{df["label"].value_counts().to_dict()}\n"
             f"Class 0 ({class0_label}) precision={class_0_precision:.02f} sensitivity{class_0_sensitivity:.02f}\n "
             f"Class 1 ({class1_label}) precision={class_1_precision:.02f} sensitivity{class_1_sensitivity:.02f}\n ")
    show_db(out_dir, title, model, X_pca, y, X_train, y_train, X_test, y_test)
    return df


if __name__ == "__main__":
    df_samples = svm(Path("output/ml"), Path("output/datasets/cleaned_dataset_pulse.csv"))


