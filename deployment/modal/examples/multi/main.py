# This code is editable! Click the Run button below to execute.

import modal

app = modal.App("example-scaling-out")

image = modal.Image.debian_slim().pip_install("scikit-learn~=1.5.0")


@app.function(image=image)
def fit_knn(k):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    # train k-nearest neighbors classifier on digits dataset
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    clf = KNeighborsClassifier(k)
    clf.fit(X_train, y_train)
    score = float(clf.score(X_test, y_test))
    print("k = %3d, score = %.4f" % (k, score))
    return score, k


@app.local_entrypoint()
def main():
    results = []
    #for k in range(1, 100):  # TODO: parallelize this
    #    results.append(fit_knn.remote(k))
    results=fit_knn.map(range(1,100))

    best_score, best_k = max(results)
    print("Best k = %3d, score = %.4f" % (best_k, best_score))
