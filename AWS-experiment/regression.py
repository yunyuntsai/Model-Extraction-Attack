import boto3
import numpy as np
import argparse
import _pickle
from copy import copy
import decimal
import re
import timeit
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import logging
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='a dataset')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--incomplete', dest='incomplete',
                        action='store_true', help='allow incomplete queries')
    args = parser.parse_args()

    dataset = args.data
    seed = args.seed
    incomplete = args.incomplete
    verbose = args.verbose

    if verbose:
        level = logging.INFO

        logger = logging.getLogger()
        logger.setLevel(level)
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(level)
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    np.random.seed(seed)

    _, _, X, _, _ = utils.prepare_data(dataset, onehot=False, labelEncode=False)

    cat_idx = [i for i in range(len(X.columns))
               if isinstance(X.iloc[0][i], basestring)]
    cont_idx = range(X.shape[1])
    for i in cat_idx:
        cont_idx.remove(i)
    X = X[cat_idx + cont_idx].values

    ext = AWSRegressionExtractor(dataset, X.copy(), cat_idx,
                                 incomplete=incomplete)

    try:
        X_test = X[0:500]

        if ext.binning:
            r = -decimal.Decimal(str(ext.eps)).as_tuple().exponent
            for i, t in enumerate(ext.feature_types):
                if t == "NUMERIC":
                    X_test[:, i] = np.round(X_test[:, i].astype(np.float), r)
    except ValueError:
        X_test = None

    ext.run(args.data, X_test, 500, random_seed=seed,
            alphas=[1], methods=['passive'], baseline=False)

if __name__ == "__main__":
    main()