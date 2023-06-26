# import packages
import numpy as np
from numpy import random

import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import PolynomialFeatures

import time

from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.saving import load_model


class TLearner:  # TODO: comment what is what.
    def __init__(self, method):  # TODO: or maybe not give base_learners but method, i.e. : 'lasso', 'rf' or 'nn'
        self.method = method

        if method == 'rf':
            self.mu0_model = RandomForestRegressor(n_estimators=1000, max_depth=100, random_state=0)
            self.mu1_model = RandomForestRegressor(n_estimators=1000, max_depth=100, random_state=0)
        elif method == 'lasso':
            self.mu0_model = LassoCV(cv=10, tol=1e-2, random_state=0, max_iter=100000)
            self.mu1_model = LassoCV(cv=10, tol=1e-2, random_state=0, max_iter=100000)
            self.poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
        elif method == 'nn':
            self.mu0_model = load_model('model_25')
            self.mu1_model = load_model('model_25')
        else:
            raise NotImplementedError('Base learner method not specified')

    def fit(self,
            x, y, w):  # TODO: training process
        if self.method == 'rf':
            # 1: train mu_0
            print("Fitting RF for mu_0")
            self.mu0_model.fit(x[w == 0], y[w == 0])

            # 2: train mu_1
            print("Fitting RF for mu_1")
            self.mu1_model.fit(x[w == 1], y[w == 1])

        elif self.method == 'lasso':
            # make polynomial features
            x_poly_train = self.poly.fit_transform(x)

            # 1: train mu_0
            print("Fitting Lasso for mu_0")
            self.mu0_model.fit(x_poly_train[w == 0], y[w == 0])

            # 2: train mu_1
            print("Fitting Lasso for mu_1")
            self.mu1_model.fit(x_poly_train[w == 1], y[w == 1])

        elif self.method == 'nn':
            # 1: train mu_0
            print("Training NN for mu_0")
            self.mu0_model.fit(x[w == 0], y[w == 0],
                               batch_size=100,
                               epochs=100,
                               callbacks=None,  # include early stopping
                               verbose=0
                               )

            # 2: train mu_1
            print("Training NN for mu_1")
            self.mu1_model.fit(x[w == 1], y[w == 1],
                               batch_size=100,
                               epochs=100,
                               callbacks=None,  # include early stopping
                               verbose=0
                               )

        else:
            raise NotImplementedError('Base learner method not specified in fit')

    def predict(self,
                x):  # TODO:
        if self.method == 'rf':
            # 1: calculate hats of mu_1 & mu_0
            mu0_hats = self.mu0_model.predict(x)
            mu1_hats = self.mu1_model.predict(x)
            predictions = mu1_hats - mu0_hats

        elif self.method == 'lasso':
            # make polynomial features
            x_poly_test = self.poly.fit_transform(x)

            # 1: calculate hats of mu_1 & mu_0
            mu0_hats = self.mu0_model.predict(x_poly_test)
            mu1_hats = self.mu1_model.predict(x_poly_test)
            predictions = mu1_hats - mu0_hats

        elif self.method == 'nn':
            mu0_hats = self.mu0_model.predict(x, verbose=0)
            mu1_hats = self.mu1_model.predict(x, verbose=0)
            predictions = np.reshape(mu1_hats - mu0_hats, (len(x),))

        else:
            raise NotImplementedError('Base learner method not specified in predict')
        return predictions


class SLearner:  # TODO: comment what is what.
    def __init__(self, method):  # TODO: or maybe not give base_learners but method, i.e. : 'lasso', 'rf' or 'nn'
        self.method = method

        if method == 'rf':
            self.mux_model = RandomForestRegressor(n_estimators=1000, max_depth=100, random_state=0)
        elif method == 'lasso':
            self.mux_model = LassoCV(cv=10, tol=1e-2, random_state=0, max_iter=100000)
            self.poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
        elif method == 'nn':
            self.mux_model = load_model('model_26')
        else:
            raise NotImplementedError('Base learner method not specified')

    def fit(self,
            x, y, w):  # TODO: training process
        x_w = np.concatenate((x, np.reshape(w, (len(w), 1))), axis=1)

        if self.method == 'rf':
            # 1: train mu_x
            print("Fitting RF for mu_x")
            self.mux_model.fit(x_w, y)

        elif self.method == 'lasso':
            # make polynomial features
            x_poly_train = self.poly.fit_transform(x_w)

            # 1: train mu_x
            print("Fitting Lasso for mu_x")
            self.mux_model.fit(x_poly_train, y)


        elif self.method == 'nn':
            # 1: train mu_x
            print("Training NN for mu_x")
            self.mux_model.fit(x_w, y,
                               batch_size=100,
                               epochs=100,
                               callbacks=None,  # include early stopping
                               verbose=0
                               )

        else:
            raise NotImplementedError('Base learner method not specified in fit')

    def predict(self,
                x):  # TODO:
        x_0 = np.concatenate((x, np.zeros((len(x), 1))), axis=1)
        x_1 = np.concatenate((x, np.ones((len(x), 1))), axis=1)

        if self.method == 'rf':
            # 1: calculate hats of mu_x with X and W=1 or W=0
            mu0_hats = self.mux_model.predict(x_0)
            mu1_hats = self.mux_model.predict(x_1)
            predictions = mu1_hats - mu0_hats

        elif self.method == 'lasso':
            # make polynomial features
            x_poly_0 = self.poly.fit_transform(x_0)
            x_poly_1 = self.poly.fit_transform(x_1)

            # 1: calculate hats of mu_x with X and W=1 or W=0
            mu0_hats = self.mux_model.predict(x_poly_0)
            mu1_hats = self.mux_model.predict(x_poly_1)
            predictions = mu1_hats - mu0_hats

        elif self.method == 'nn':
            # 1: calculate hats of mu_x with X and W=1 or W=0
            mu0_hats = self.mux_model.predict(x_0, verbose=0)
            mu1_hats = self.mux_model.predict(x_1, verbose=0)
            predictions = np.reshape(mu1_hats - mu0_hats, (len(x),))

        else:
            raise NotImplementedError('Base learner method not specified in predict')
        return predictions


class XLearner:  # TODO: comment what is what.
    def __init__(self, method):  # TODO: or maybe not give base_learners but method, i.e. : 'lasso', 'rf' or 'nn'
        self.method = method

        if method == 'rf':
            self.mu0_model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)
            self.mu1_model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)
            self.ex_model = RandomForestClassifier(max_depth=100, random_state=0)
            self.tau0_model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)
            self.tau1_model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)

        elif method == 'lasso':
            self.mu0_model = LassoCV(cv=10, tol=1, random_state=0, max_iter=100000)
            self.mu1_model = LassoCV(cv=10, tol=1, random_state=0, max_iter=100000)
            self.ex_model = LogisticRegressionCV(cv=KFold(10), penalty='l1', solver='saga', tol=1, random_state=0,
                                                 max_iter=100000)
            self.tau0_model = LassoCV(cv=10, tol=1, random_state=0, max_iter=100000)
            self.tau1_model = LassoCV(cv=10, tol=1, random_state=0, max_iter=100000)
            self.poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)

        elif method == 'nn':
            self.mu0_model = load_model('model_25')
            self.mu1_model = load_model('model_25')
            self.ex_model = load_model('model_ex')
            self.tau0_model = load_model('model_25')
            self.tau1_model = load_model('model_25')
        else:
            raise NotImplementedError('Base learner method not specified')

    def fit(self,
            x, y, w):  # TODO: training process
        if self.method == 'rf':
            # 1: train mu_0 and get imputed_1
            print("Fitting RF for mu_0")
            self.mu0_model.fit(x[w == 0], y[w == 0])
            imputed_1 = y[w == 1] - self.mu0_model.predict(x[w == 1])

            # 2: train mu_1 and get imputed_0
            print("Fitting RF for mu_1")
            self.mu1_model.fit(x[w == 1], y[w == 1])
            imputed_0 = self.mu1_model.predict(x[w == 0]) - y[w == 0]

            # 3: train tau_0
            print("Fitting RF for tau_0")
            self.tau0_model.fit(x[w == 0], imputed_0)

            # 4: train tau_1
            print("Fitting RF for tau_1")
            self.tau1_model.fit(x[w == 1], imputed_1)

            # 5: train e_x
            print("Fitting RF for e_x")
            self.ex_model.fit(x, w)

        elif self.method == 'lasso':
            # make polynomial features
            x_poly_train = self.poly.fit_transform(x)

            # 1: train mu_0 and get imputed_1
            print("Fitting Lasso for mu_0")
            self.mu0_model.fit(x_poly_train[w == 0], y[w == 0])
            imputed_1 = y[w == 1] - self.mu0_model.predict(x_poly_train[w == 1])

            # 2: train mu_1 and get imputed_0
            print("Fitting Lasso for mu_1")
            self.mu1_model.fit(x_poly_train[w == 1], y[w == 1])
            imputed_0 = self.mu1_model.predict(x_poly_train[w == 0]) - y[w == 0]

            # 3: train tau_0
            print("Fitting Lasso for tau_0")
            self.tau0_model.fit(x_poly_train[w == 0], imputed_0)

            # 4: train tau_1
            print("Fitting Lasso for tau_1")
            self.tau1_model.fit(x_poly_train[w == 1], imputed_1)

            # 5: train e_x
            print("Fitting Lasso for e_x")
            self.ex_model.fit(x_poly_train, w)

        elif self.method == 'nn':
            # 1: train mu_0
            print("Training NN for mu_0")
            self.mu0_model.fit(x[w == 0], y[w == 0],
                               batch_size=100,
                               epochs=100,
                               callbacks=None,  # include early stopping
                               verbose=0
                               )
            imputed_1 = y[w == 1] - np.reshape(self.mu0_model.predict(x[w == 1], verbose=0), (len(x[w == 1]),))

            # 2: train mu_1
            print("Training NN for mu_1")
            self.mu1_model.fit(x[w == 1], y[w == 1],
                               batch_size=100,
                               epochs=100,
                               callbacks=None,  # include early stopping
                               verbose=0
                               )
            imputed_0 = np.reshape(self.mu1_model.predict(x[w == 0], verbose=0), (len(x[w == 0]),)) - y[w == 0]

            # 3: train tau_0
            print("Fitting NN for tau_0")
            self.tau0_model.fit(x[w == 0], imputed_0,
                                batch_size=100,
                                epochs=100,
                                callbacks=None,  # include early stopping
                                verbose=0
                                )

            # 4: train tau_1
            print("Fitting NN for tau_1")
            self.tau1_model.fit(x[w == 1], imputed_1,
                                batch_size=100,
                                epochs=100,
                                callbacks=None,  # include early stopping
                                verbose=0
                                )

            # 5: train e_x
            print("Fitting NN for e_x")
            self.ex_model.fit(x, w,
                              batch_size=100,
                              epochs=100,
                              callbacks=None,  # include early stopping
                              verbose=0
                              )

        else:
            raise NotImplementedError('Base learner method not specified in fit')

    def predict(self,
                x):
        if self.method == 'rf':
            # 1: calculate hats of tau_0 and tau_1
            tau_0_hats = self.tau0_model.predict(x)
            tau_1_hats = self.tau1_model.predict(x)
            # 2: probabilities
            probs = self.ex_model.predict_proba(x)[:, 1]
            # 3: final predictions

        elif self.method == 'lasso':
            # make polynomial features
            x_poly_test = self.poly.fit_transform(x)

            # 1: calculate hats of tau_0 and tau_1
            tau_0_hats = self.tau0_model.predict(x_poly_test)
            tau_1_hats = self.tau1_model.predict(x_poly_test)
            probs = self.ex_model.predict_proba(x_poly_test)[:, 1]

        elif self.method == 'nn':
            # 1: calculate hats of tau_0 and tau_1
            tau_0_hats = np.reshape(self.tau0_model.predict(x, verbose=0), (len(x),))
            tau_1_hats = np.reshape(self.tau1_model.predict(x, verbose=0), (len(x),))
            # 2: probabilities
            logit = self.ex_model.predict(x, verbose=0)
            probs = np.reshape(keras.activations.sigmoid(logit), (len(logit, )))
            # 3: final predictions

        else:
            raise NotImplementedError('Base learner method not specified in predict')

        predictions = probs * tau_0_hats + (1 - probs) * tau_1_hats
        return predictions


class RLearner:
    def __init__(self, method):
        self.method = method

        if method == 'rf':
            self.mux_model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)
            self.ex_model = RandomForestClassifier(max_depth=100, random_state=0)
            self.tau_model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)

        elif method == 'lasso':
            self.mux_model = LassoCV(cv=10, tol=1, random_state=0, max_iter=100000)
            self.ex_model = LogisticRegressionCV(cv=KFold(10), penalty='l1', solver='saga', tol=1, random_state=0,
                                                 max_iter=100000)
            self.tau_model = LassoCV(cv=10, tol=1, random_state=0, max_iter=100000)
            self.poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)

        elif method == 'nn':
            self.mux_model = load_model('model_25')
            self.ex_model = load_model('model_ex')
            self.tau_model = load_model('model_25')

        else:
            raise NotImplementedError('Base learner method not specified or typo')

    def fit(self, x, y, w):

        if self.method == 'rf':
            # 1: fit mu_x
            print('Fitting RF for mu_x')
            self.mux_model.fit(x, y)

            print('Fitting RF for e_x')
            # 2: fit ex
            self.ex_model.fit(x, w)

            # 3: calculate pseudo_outcomes & weights
            probs = self.ex_model.predict_proba(x)[:, 1]
            pseudo_outcomes = (y - self.mux_model.predict(x)) / (w - probs + 0.01)  # TODO: change these!!!
            weights = (w - probs) ** 2

            print('Fitting RF for tau_x')
            # 4: fit tau
            self.tau_model.fit(x, pseudo_outcomes, sample_weight=weights)

        elif self.method == 'lasso':
            x_poly_train = self.poly.fit_transform(x)

            # 1: fit mu_x
            print('Fitting Lasso for mu_x')
            self.mux_model.fit(x_poly_train, y)

            # 2: fit ex
            print('Fitting Lasso for e_x')
            self.ex_model.fit(x_poly_train, w)

            # 3: calculate pseudo_outcomes & weights
            probs = self.ex_model.predict_proba(x_poly_train)[:, 1]
            pseudo_outcomes = (y - self.mux_model.predict(x_poly_train)) / (w - probs + 0.01)
            weights = (w - probs) ** 2

            # 4: fit tau
            print('Fitting Lasso for tau_x')
            self.tau_model.fit(x_poly_train, pseudo_outcomes, sample_weight=weights)

        elif self.method == 'nn':

            # 1: fit mu_x
            print('Training NN for mu_x')
            self.mux_model.fit(x, y,
                               batch_size=100,
                               epochs=100,
                               callbacks=None,
                               verbose=0
                               )
            # 2: fit ex
            print('Training NN for e_x')
            self.ex_model.fit(x, w,
                              batch_size=100,
                              epochs=100,
                              callbacks=None,
                              verbose=0
                              )

            # 3: calculate pseudo_outcomes & weights
            probs = np.reshape(keras.activations.sigmoid(self.ex_model.predict(x, verbose=0)), len(x, ))
            pseudo_outcomes = (y - np.reshape(self.mux_model.predict(x, verbose=0), (len(x),))) / (w - probs + 0.01)
            weights = (w - probs) ** 2

            # 4: fit tau
            print('Training NN for tau_x')
            self.tau_model.fit(x, pseudo_outcomes,
                               sample_weight=weights,
                               batch_size=100,
                               epochs=100,
                               validation_data=None,
                               callbacks=None,
                               verbose=0
                               )

        else:
            raise NotImplementedError('Base learner method not specified in fit')

    def predict(self, x):

        if self.method == 'rf':
            predictions = self.tau_model.predict(x)

        elif self.method == 'lasso':
            x_poly_test = self.poly.fit_transform(x)
            predictions = self.tau_model.predict(x_poly_test)

        elif self.method == 'nn':
            predictions = np.reshape(self.tau_model.predict(x, verbose=0), (len(x),))

        else:
            raise NotImplementedError('Base learner method not specified in predict')

        return predictions


class DRLearner:
    def __init__(self, method):
        self.method = method
        if method == 'rf':
            self.mu0_model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)
            self.mu1_model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)
            self.ex_model = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=0)
            self.tau_model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)

        elif method == 'lasso':
            self.mu0_model = LassoCV(cv=10, tol=1, random_state=0, max_iter=100000)
            self.mu1_model = LassoCV(cv=10, tol=1, random_state=0, max_iter=100000)
            self.ex_model = LogisticRegressionCV(cv=KFold(10), penalty='l1', solver='saga', tol=1, random_state=0,
                                                 max_iter=100000)
            self.tau_model = LassoCV(cv=10, tol=1, random_state=0, max_iter=100000)
            self.poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)

        elif method == 'nn':
            self.mu0_model = load_model('model_25')
            self.mu1_model = load_model('model_25')
            self.ex_model = load_model('model_ex')
            self.tau_model = load_model('model_25')

        else:
            raise NotImplementedError('Base learner method not specified or typo')

    def fit(self, x, y, w):

        if self.method == 'rf':
            # 1: fit mu_0
            print('Fitting RF for mu_0')
            self.mu0_model.fit(x[w == 0], y[w == 0])

            # 2: fit mu_1
            print('Fitting RF for mu_1')
            self.mu1_model.fit(x[w == 1], y[w == 1])

            # 3: fit ex
            print('Fitting RF for e_x')
            self.ex_model.fit(x, w)
            probs = self.ex_model.predict_proba(x)[:, 1]
            neg_prob = self.ex_model.predict_proba(x)[:, 0]

            # calculate pseudo_outcomes
            mu_w = w * self.mu1_model.predict(x) + (1 - w) * self.mu0_model.predict(x)
            pseudo_outcomes = (w - probs) / (probs * neg_prob + 0.01) * (y - mu_w) + self.mu1_model.predict(
                x) - self.mu0_model.predict(x)  # TODO: CHANGE THESE 0.01s!

            # 4 fit tau
            print('Fitting RF for tau_x')
            self.tau_model.fit(x, pseudo_outcomes)

        elif self.method == 'lasso':
            x_poly_train = self.poly.fit_transform(x)

            # 1: fit mu_0
            print('Fitting lasso for mu_0')
            self.mu0_model.fit(x_poly_train[w == 0], y[w == 0])

            # 2: fit mu_1
            print('Fitting lasso for mu_1')
            self.mu1_model.fit(x_poly_train[w == 1], y[w == 1])

            # 3: fit ex
            print('Fitting lasso for e_x')
            self.ex_model.fit(x_poly_train, w)
            probs = self.ex_model.predict_proba(x_poly_train)[:, 1]

            # calculate pseudo_outcomes
            mu_w = w * self.mu1_model.predict(x_poly_train) + (1 - w) * self.mu0_model.predict(x_poly_train)
            pseudo_outcomes = (w - probs) / (probs * (1 - probs) + 0.01) * (y - mu_w) + self.mu1_model.predict(
                x_poly_train) - self.mu0_model.predict(x_poly_train)

            # 4 fit tau
            print('Fitting lasso for tau_x')
            self.tau_model.fit(x_poly_train, pseudo_outcomes)

        elif self.method == 'nn':

            # 1: fit mu_0
            print('Training NN for mu_0')
            self.mu0_model.fit(x[w == 0], y[w == 0],
                               batch_size=100,
                               epochs=100,
                               callbacks=None,
                               verbose=0
                               )

            # 2: fit mu_1
            print('Training NN for mu_1')
            self.mu1_model.fit(x[w == 1], y[w == 1],
                               batch_size=100,
                               epochs=100,
                               callbacks=None,
                               verbose=0
                               )

            # 3: fit ex
            print('Training NN for e_x')
            self.ex_model.fit(x, w,
                              batch_size=100,
                              epochs=100,
                              callbacks=None,
                              verbose=0
                              )

            probs = np.reshape(keras.activations.sigmoid(self.ex_model.predict(x, verbose=0)), len(x, ))

            # calculate pseudo_outcomes
            mu_w = w * self.mu1_model.predict(x, verbose=0) + (1 - w) * self.mu0_model.predict(x, verbose=0)
            pseudo_outcomes = (w - probs) / (probs * (1 - probs) + 0.01) * (y - mu_w) + self.mu1_model.predict(x,
                                                                                                        verbose=0) - self.mu0_model.predict(
                x, verbose=0)

            # 4 fit tau
            print('Training NN for tau_x')
            self.tau_model.fit(x, pseudo_outcomes,
                               batch_size=100,
                               epochs=100,
                               validation_data=None,
                               callbacks=None,
                               verbose=0
                               )

    def predict(self, x):

        if self.method == 'rf':
            predictions = self.tau_model.predict(x)


        elif self.method == 'lasso':
            x_poly_test = self.poly.fit_transform(x)
            predictions = self.tau_model.predict(x_poly_test)

        elif self.method == 'nn':
            predictions = np.reshape(self.tau_model.predict(x, verbose=0), (len(x),))

        else:
            raise NotImplementedError('Base learner method not specified in predict')

        return predictions


class RALearner:
    def __init__(self, method):
        self.method = method
        if method == 'rf':
            self.mu0_model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)
            self.mu1_model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)
            self.tau_model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)

        elif method == 'lasso':
            self.mu0_model = LassoCV(cv=10, tol=1, random_state=0, max_iter=100000)
            self.mu1_model = LassoCV(cv=10, tol=1, random_state=0, max_iter=100000)
            self.tau_model = LassoCV(cv=10, tol=1, random_state=0, max_iter=100000)
            self.poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)

        elif method == 'nn':
            self.mu0_model = load_model('model_25')
            self.mu1_model = load_model('model_25')
            self.tau_model = load_model('model_25')

        else:
            raise NotImplementedError('Base learner method not specified or typo')

    def fit(self, x, y, w):
        if self.method == 'rf':
            # 1: fit mu_0
            print('Fitting RF for mu_0')
            self.mu0_model.fit(x[w == 0], y[w == 0])

            # 2: fit mu_1
            print('Fitting RF for mu_1')
            self.mu1_model.fit(x[w == 1], y[w == 1])

            # calculate pseudo_outcomes
            pseudo_outcomes = w * (y - self.mu0_model.predict(x)) + (1 - w) * (self.mu1_model.predict(x) - y)

            # 4 fit tau
            print('Fitting RF for tau_x')
            self.tau_model.fit(x, pseudo_outcomes)

        elif self.method == 'lasso':
            x_poly_train = self.poly.fit_transform(x)

            # 1: fit mu_0
            print('Fitting Lasso for mu_0')
            self.mu0_model.fit(x_poly_train[w == 0], y[w == 0])

            # 2: fit mu_1
            print('Fitting Lasso for mu_1')
            self.mu1_model.fit(x_poly_train[w == 1], y[w == 1])

            # calculate pseudo_outcomes
            pseudo_outcomes = w * (y - self.mu0_model.predict(x_poly_train)) + (1 - w) * (
                    self.mu1_model.predict(x_poly_train) - y)

            # 4 fit tau
            print('Fitting Lasso for tau_x')
            self.tau_model.fit(x_poly_train, pseudo_outcomes)

        elif self.method == 'nn':

            # 1: fit mu_0
            print('Training NN for mu_0')
            self.mu0_model.fit(x[w == 0], y[w == 0],
                               batch_size=100,
                               epochs=100,
                               callbacks=None,
                               verbose=0
                               )

            # 2: fit mu_1
            print('Training NN for mu_1')
            self.mu1_model.fit(x[w == 1], y[w == 1],
                               batch_size=100,
                               epochs=100,
                               callbacks=None,
                               verbose=0
                               )

            # calculate pseudo_outcomes
            mu0_predictions = np.reshape(self.mu0_model.predict(x, verbose=0), (len(x),))
            mu1_predictions = np.reshape(self.mu1_model.predict(x, verbose=0), (len(x),))

            pseudo_outcomes = w * (y - mu0_predictions) + (1 - w) * (mu1_predictions - y)

            # 4 fit tau
            print('Training NN for tau_x')
            self.tau_model.fit(x, pseudo_outcomes,
                               batch_size=100,
                               epochs=100,
                               validation_data=None,
                               callbacks=None,
                               verbose=0
                               )

    def predict(self, x):
        if self.method == 'rf':
            predictions = self.tau_model.predict(x)


        elif self.method == 'lasso':
            x_poly_test = self.poly.fit_transform(x)
            predictions = self.tau_model.predict(x_poly_test)

        elif self.method == 'nn':
            predictions = np.reshape(self.tau_model.predict(x, verbose=0), (len(x),))

        else:
            raise NotImplementedError('Base learner method not specified in predict')

        return predictions


class PWLearner:
    def __init__(self, method):
        self.method = method
        if method == 'rf':
            self.ex_model = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=0)
            self.tau_model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)

        elif method == 'lasso':
            self.ex_model = LogisticRegressionCV(cv=KFold(10), penalty='l1', solver='saga', tol=1, random_state=0,
                                                 max_iter=100000)
            self.tau_model = LassoCV(cv=10, tol=1, random_state=0, max_iter=100000)
            self.poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)

        elif method == 'nn':
            self.ex_model = load_model('model_ex')
            self.tau_model = load_model('model_25')

        else:
            raise NotImplementedError('Base learner method not specified or typo')

    def fit(self, x, y, w):

        if self.method == 'rf':
            # 3: fit ex
            print('Fitting RF for e_x')
            self.ex_model.fit(x, w)
            probs = self.ex_model.predict_proba(x)[:, 1]
            counter_probs = self.ex_model.predict_proba(x)[:, 0]

            # calculate pseudo_outcomes
            pseudo_outcomes = (w / (probs + 0.01) - (1 - w) / (counter_probs + 0.01)) * y  # TODO: CHANGE 0.01!

            # 4 fit tau
            print('Fitting RF for tau_x')
            self.tau_model.fit(x, pseudo_outcomes)

        elif self.method == 'lasso':
            x_poly_train = self.poly.fit_transform(x)

            # 3: fit ex
            print('Fitting Lasso for e_x')
            self.ex_model.fit(x_poly_train, w)

            probs = self.ex_model.predict_proba(x_poly_train)[:, 1]
            counter_probs = self.ex_model.predict_proba(x_poly_train)[:, 0]

            # calculate pseudo_outcomes
            pseudo_outcomes = (w / (probs + 0.01) - (1 - w) / (counter_probs + 0.01)) * y

            # 4 fit tau
            print('Fitting Lasso for tau_x')
            self.tau_model.fit(x_poly_train, pseudo_outcomes)

        elif self.method == 'nn':

            # 3: fit ex
            print('Training NN for e_x')
            self.ex_model.fit(x, w,
                              batch_size=100,
                              epochs=100,
                              callbacks=None,
                              verbose=0
                              )

            probs = np.reshape(keras.activations.sigmoid(self.ex_model.predict(x, verbose=0)), len(x, ))
            counter_probs = 1 - probs

            # calculate pseudo_outcomes
            pseudo_outcomes = (w / (probs + 0.01) - (1 - w) / (counter_probs + 0.01)) * y

            # 4 fit tau
            print('Training NN for tau_x')
            self.tau_model.fit(x, pseudo_outcomes,
                               batch_size=100,
                               epochs=100,
                               validation_data=None,
                               callbacks=None,
                               verbose=0
                               )

    def predict(self, x):
        if self.method == 'rf':
            predictions = self.tau_model.predict(x)

        elif self.method == 'lasso':
            x_poly_test = self.poly.fit_transform(x)
            predictions = self.tau_model.predict(x_poly_test)

        elif self.method == 'nn':
            predictions = np.reshape(self.tau_model.predict(x, verbose=0), (len(x),))

        else:
            raise NotImplementedError('Base learner method not specified in predict')

        return predictions


class ULearner:
    def __init__(self, method):
        self.method = method
        if method == 'rf':
            self.mux_model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)
            self.ex_model = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=0)
            self.tau_model = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)

        elif method == 'lasso':
            self.mux_model = LassoCV(cv=10, tol=1, random_state=0, max_iter=100000)
            self.ex_model = LogisticRegressionCV(cv=KFold(10), penalty='l1', solver='saga', tol=1, random_state=0,
                                                 max_iter=100000)
            self.tau_model = LassoCV(cv=10, tol=1, random_state=0, max_iter=100000)
            self.poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)

        elif method == 'nn':
            self.mux_model = load_model('model_25')
            self.ex_model = load_model('model_ex')
            self.tau_model = load_model('model_25')

        else:
            raise NotImplementedError('Base learner method not specified or typo')

    def fit(self, x, y, w):

        if self.method == 'rf':
            # 2: fit mu_x
            print('Fitting RF for mu_x')
            self.mux_model.fit(x, y)

            # 3: fit ex
            print('Fitting RF for e_x')
            self.ex_model.fit(x, w)
            probs = self.ex_model.predict_proba(x)[:, 1]

            # calculate residuals
            residuals = (y - self.mux_model.predict(x)) / (w - probs + 0.01)  # TODO: CHANGE 0.01

            # 4 fit tau
            print('Fitting RF for tau_x')
            self.tau_model.fit(x, residuals)

        elif self.method == 'lasso':
            x_poly_train = self.poly.fit_transform(x)

            # 2: fit mu_x
            print('Fitting Lasso for mu_x')
            self.mux_model.fit(x_poly_train, y)

            # 3: fit ex
            print('Fitting Lasso for e_x')
            self.ex_model.fit(x_poly_train, w)
            probs = self.ex_model.predict_proba(x_poly_train)[:, 1]

            # calculate pseudo_outcomes
            residuals = (y - self.mux_model.predict(x_poly_train)) / (w - probs + 0.01)

            # 4 fit tau
            print('Fitting Lasso for tau_x')
            self.tau_model.fit(x_poly_train, residuals)

        elif self.method == 'nn':

            # 1: fit mu_x
            print('Training NN for mu_x')
            self.mux_model.fit(x, y,
                               batch_size=100,
                               epochs=100,
                               callbacks=None,
                               verbose=0
                               )

            # 3: fit ex
            print('Training NN for e_x')
            self.ex_model.fit(x, w,
                              batch_size=100,
                              epochs=100,
                              callbacks=None,
                              verbose=0
                              )

            probs = np.reshape(keras.activations.sigmoid(self.ex_model.predict(x, verbose=0)), len(x, ))

            # calculate pseudo_outcomes
            mu_x_predictions = np.reshape(self.mux_model.predict(x, verbose=0), (len(x),))
            residuals = (y - mu_x_predictions) / (w - probs + 0.01)

            # 4 fit tau
            print('Training NN for tau_x')
            self.tau_model.fit(x, residuals,
                               batch_size=100,
                               epochs=100,
                               validation_data=None,
                               callbacks=None,
                               verbose=0
                               )

    def predict(self, x):

        if self.method == 'rf':
            predictions = self.tau_model.predict(x)

        elif self.method == 'lasso':
            x_poly_test = self.poly.fit_transform(x)
            predictions = self.tau_model.predict(x_poly_test)

        elif self.method == 'nn':
            predictions = np.reshape(self.tau_model.predict(x, verbose=0), (len(x),))

        else:
            raise NotImplementedError('Base learner method not specified in predict')

        return predictions
