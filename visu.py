import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import LogisticRegression

# Set matplotlib to use Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
from sklearn import svm


def show_db(out_dir, score, clf, X, y, X_train, y_train, X_test, y_test, df_meta):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    fig, ax = plt.subplots()
    fontsize = 8

    is_linear_model = False
    if hasattr(clf, 'kernel'):
        if clf.kernel == 'linear':
            is_linear_model = True

    if isinstance(clf, LogisticRegression):
        is_linear_model = True

    if is_linear_model:
        # get the separating hyperplane
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace(-500, 500)
        yy = a * xx - (clf.intercept_[0]) / w[1]

        # plot the parallels to the separating hyperplane that pass through the
        # support vectors (margin away from hyperplane in direction
        # perpendicular to hyperplane). This is sqrt(1+a^2) away vertically in
        # 2-d.
        margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
        yy_down = yy - np.sqrt(1 + a ** 2) * margin
        yy_up = yy + np.sqrt(1 + a ** 2) * margin

        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )
        # Plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )
        # Plot the testing points
        colors = {0 : "r", 1: 'b'}
        for g in [0, 1]:
            ax.scatter(
                X_test[y_test == g][:, 0],
                X_test[y_test == g][:, 1],
                c=colors[g],
                cmap=cm_bright,
                edgecolors="yellow",
                alpha=0.6,
                label=f"Class {g}"
            )
        plt.plot(xx, yy, "k-")
        plt.plot(xx, yy_down, "k--")
        plt.plot(xx, yy_up, "k--")
        plt.legend()

        # Annotate points with metadata
        for i in range(len(X)):
            ax.annotate(df_meta.iloc[i], (X[i, 0], X[i, 1]), ha='right', fontsize=fontsize)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(score)
        fig.tight_layout()
    else:
        DecisionBoundaryDisplay.from_estimator(
            clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5
        )
        # Plot the training points
        ax.scatter(
            X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
        )

        # Plot the testing points
        colors = {0 : "r", 1: 'b'}
        for g in [0, 1]:
            ax.scatter(
                X_test[y_test == g][:, 0],
                X_test[y_test == g][:, 1],
                c=colors[g],
                #cmap=cm_bright,
                edgecolors="yellow",
                alpha=0.6,
                label=f"Class {g}"
            )

        # plot db lines:
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.contour(
            XX,
            YY,
            Z,
            colors=["k", "k", "k"],
            linestyles=["--", "-", "--"],
            levels=[-0.5, 0, 0.5],
        )

        # Annotate points with metadata
        for i in range(len(X)):
            ax.annotate(df_meta.iloc[i], (X[i, 0], X[i, 1]), ha='right', fontsize=fontsize)

        ax.set_title(score)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend()
        fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.show()
    filepath = out_dir / "db.png"
    print(filepath)
    fig.savefig(filepath, dpi=400)
