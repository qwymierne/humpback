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


def example_1(repeats=10):
    signal_space = np.linspace(0.01, 5.01, 50)
    aic = AIC(LinearRegression())
    bic = BIC(LinearRegression())
    mbic = mBIC(LinearRegression(), 0.1, 0.01)
    mbic2 = mBIC2(LinearRegression())
    ics = {'aic': aic, 'bic': bic, 'mbic': mbic, 'mbic2': mbic2}
    false_positives_ratio = {'aic': [], 'bic': [], 'mbic': [], 'mbic2': []}
    false_negatives_ratio = {'aic': [], 'bic': [], 'mbic': [], 'mbic2': []}
    for signal in tqdm(signal_space):
        for ic in ics:
            fprs, fnrs = [], []
            for _ in range(repeats):
                X = np.random.randn(100, 95)
                beta = np.array([signal] * 10 + [0] * 85).reshape([-1, 1])
                eps = np.random.randn(100, 1)
                y = (X @ beta + eps).reshape(-1)
                cs = ColumnsSelector(ics[ic], LassoPathHeuristic())
                cs.fit(X, y)
                fprs.append(sum(cs.chosen_columns_[10:]) / 85)
                fnrs.append(1 - sum(cs.chosen_columns_[:10]) / 10)
            false_positives_ratio[ic].append(sum(fprs) / repeats)
            false_negatives_ratio[ic].append(sum(fnrs) / repeats)

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


def show_important_pixels(mask, vals, label):
    base = np.zeros(784) + 127
    vi = iter(vals)
    for i, m in enumerate(mask):
        if m == 1.:
            vv = next(vi)
            base[i] = 255 if vv > 0 else 0
    plt.imshow(base.reshape((28, 28)), cmap='gray')
    plt.title(f'Key pixels for {label}')
    plt.show()


def example_2():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X[:5000]
    y = y[:5000]

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mbic2_cs', ColumnsSelector(mBIC2(LinearRegression(fit_intercept=False)),
                                     LassoPathHeuristic(fit_intercept=False),
                                     interactions=False))])
    for digit in '0123456789':
        yd = np.where(y == digit, 1., 0.)
        pipe.fit(X, yd)
        cc = pipe['mbic2_cs'].chosen_columns_
        X_trans = pipe.transform(X)
        lr = LinearRegression().fit(X_trans, yd)
        coefs = lr.coef_
        show_important_pixels(cc, coefs, digit)

        Xs = X[yd == 1., :]
        avg_X = np.mean(Xs, axis=0)
        plt.imshow(avg_X.reshape((28, 28)), cmap='gray')
        plt.title(f'Average {digit}')
        plt.show()


if __name__ == '__main__':
    example_1(10)
    # example_2()
