import numpy as np

def get_k_folds(n, k=5):
    """
    Get k folds of indices for a dataset of size n.

    # Parameters
    #### n : int
        The size of the dataset.
    #### k : int
        The number of folds.

    # Returns
    A list of k tuples, where each tuple contains two arrays of indices.
    """
    idx = np.random.permutation(n)
    fold_sizes = np.full((k), [n//k])
    if n % k != 0:
        fold_sizes[:n % k] += 1
    test_idxs = folds = np.split(idx, np.cumsum(fold_sizes)[:-1])
    train_idxs = [np.setdiff1d(idx, test_idx) for test_idx in test_idxs]
    return zip(train_idxs, test_idxs)

def k_fold(X, y, fit, metric, k=5):
    """
    Perform k-fold cross-validation on a dataset.

    # Parameters
    #### X : array-like
        The dataset.
    #### y : array-like
        The labels.
    #### fit : function
        The function to fit the model.
    #### k : int
        The number of folds.

    # Returns
    An array of metrics
    """
    n = len(X)
    metrics = np.empty((n))
    for train_idx, test_idx in get_k_folds(n, k):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        model = fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics[test_idx] = metric(y_pred, y_test)
    return metrics


def bootstrap(values, reduce=np.mean, bootstraps=10000, return_values=False):
    """
    Perform bootstrapping on a list of values.

    # Parameters
    #### values : array-like
        The values to bootstrap.
    #### reduce : function
        The function to apply to the bootstrapped values.
    #### bootstraps : int
        The number of bootstraps to perform.
    #### return_values : bool
        Whether to return the bootstrapped values.

    # Returns
    The mean and standard deviation of the bootstrapped values.
    If return_values is True, the bootstrapped values are returned instead.
    """
    n = len(values)
    bootstrapped_values = np.empty((bootstraps))
    for i in range(bootstraps):
        idx = np.random.randint(0, n, n)
        bootstrapped_values[i] = reduce(values[idx])
    if return_values:
        return bootstrapped_values
    return bootstrapped_values.mean(), bootstrapped_values.std()