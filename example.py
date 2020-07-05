import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from humpback.heuristic import LassoPathHeuristic
from humpback.information_criterion import AIC, BIC, mBIC, mBIC2
from humpback.columns_selector import ColumnsSelector
import humpback.mle_extetension


def example_1():
    signal_space = np.linspace(0.01, 3.01, 50)
    aic = AIC(LinearRegression())
    bic = BIC(LinearRegression())
    mbic = mBIC(LinearRegression(), 10, 95)
    mbic2 = mBIC2(LinearRegression())
    ics = {'aic': aic, 'bic': bic, 'mbic': mbic, 'mbic2': mbic2}
    false_positives_ratio = {'aic': [], 'bic': [], 'mbic': [], 'mbic2': []}
    false_negatives_ratio = {'aic': [], 'bic': [], 'mbic': [], 'mbic2': []}
    for signal in tqdm(signal_space):
        X = np.random.randn(100, 95)
        beta = np.array([signal] * 10 + [0] * 85).reshape([-1, 1])
        eps = np.random.randn(100, 1)
        y = (X @ beta + eps).reshape(-1)

        for ic in ics:
            cs = ColumnsSelector(ics[ic], LassoPathHeuristic())
            cs.fit(X, y)
            false_positives_ratio[ic].append(sum(cs.chosen_columns_[10:]) / 85)
            false_negatives_ratio[ic].append(1 - sum(cs.chosen_columns_[:10]) / 10)

    for ic in false_positives_ratio:
        plt.plot(signal_space, false_positives_ratio[ic], label=ic)
    plt.title('False Positives Ratio')
    plt.legend()
    plt.show()

    for ic in false_negatives_ratio:
        plt.plot(signal_space, false_negatives_ratio[ic], label=ic)
    plt.title('False Negatives Ratio')
    plt.legend()
    plt.show()


def example_2():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X[:5000]
    y = y[:5000]

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('aic_cs', ColumnsSelector(AIC(LinearRegression(fit_intercept=True)),
                                   LassoPathHeuristic(fit_intercept=True),
                                   interactions=False))])

    for digit in '0123456789':
        yd = np.where(y == digit, 1., 0.)
        pipe.fit(X, yd)
        cc = pipe['aic_cs'].chosen_columns_

        plt.imshow(cc.reshape((28, 28)), cmap='gray')
        plt.title(f'Key pixels for {digit}')
        plt.show()

        Xs = X[yd == 1., :]
        avg_X = np.mean(Xs, axis=0)
        plt.imshow(avg_X.reshape((28, 28)), cmap='gray')
        plt.title(f'Average {digit}')
        plt.show()


if __name__ == '__main__':
    # example_1()
    example_2()
