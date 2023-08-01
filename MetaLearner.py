# import packages
import tensorflow as tf
import keras
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
# import from files
from neural_networks import clone_nn_regression, clone_nn_classification, NN_SEQUENTIAL, CALLBACK
from default_parameters import *

class TLearner:  # TODO: comment what is what.
    def __init__(self, method):
        self.name = "TLearner"
        self.method = method

        if self.method == 'rf':
            self.mu0_model = RandomForestRegressor(n_estimators=N_TREES,
                                                   max_depth=MAX_DEPTH,
                                                   random_state=RANDOM,
                                                   max_features=MAX_FEATURES)
            self.mu1_model = RandomForestRegressor(n_estimators=N_TREES,
                                                   max_depth=MAX_DEPTH,
                                                   random_state=RANDOM,
                                                   max_features=MAX_FEATURES)
        elif self.method == 'lasso':
            self.mu0_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.mu1_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.poly = PolynomialFeatures(degree=DEGREE_POLYNOMIALS, interaction_only=False, include_bias=False)

        elif self.method == 'nn':
            self.mu0_model = clone_nn_regression(NN_SEQUENTIAL)
            self.mu1_model = clone_nn_regression(NN_SEQUENTIAL)

        else:
            raise NotImplementedError('Base learner method not or not correctly specified')

    def fit(self,
            x, y, w):

        if self.method == 'rf':
            # 1: train mu_0
            self.mu0_model.fit(x[w == 0], y[w == 0])
            # 2: train mu_1
            self.mu1_model.fit(x[w == 1], y[w == 1])

        elif self.method == 'lasso':
            # make polynomial features
            x_poly_train = self.poly.fit_transform(x)
            # 1: train mu_0
            self.mu0_model.fit(x_poly_train[w == 0], y[w == 0])
            # 2: train mu_1
            self.mu1_model.fit(x_poly_train[w == 1], y[w == 1])

        elif self.method == 'nn':
            # to tensor
            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)
            w = tf.convert_to_tensor(w)
            # 1: train mu_0
            self.mu0_model.fit(x[w == 0], y[w == 0],
                               batch_size=BATCH_SIZE,
                               epochs=N_EPOCHS,
                               callbacks=CALLBACK,
                               validation_split=VALIDATION_SPLIT,
                               verbose=0
                               )
            # 2: train mu_1
            self.mu1_model.fit(x[w == 1], y[w == 1],
                               batch_size=BATCH_SIZE,
                               epochs=N_EPOCHS,
                               callbacks=CALLBACK,
                               validation_split=VALIDATION_SPLIT,
                               verbose=0
                               )

        else:
            raise NotImplementedError('Base learner method not specified')

    def predict(self,
                x):

        if self.method == 'rf':
            # predict
            mu0_hats = self.mu0_model.predict(x)
            mu1_hats = self.mu1_model.predict(x)
            predictions = mu1_hats - mu0_hats

        elif self.method == 'lasso':
            # make polynomial features
            x_poly_test = self.poly.fit_transform(x)
            # predict
            mu0_hats = self.mu0_model.predict(x_poly_test)
            mu1_hats = self.mu1_model.predict(x_poly_test)
            predictions = mu1_hats - mu0_hats

        elif self.method == 'nn':
            # to tensor
            x = tf.convert_to_tensor(x)
            # predict
            mu0_hats = self.mu0_model(x)
            mu1_hats = self.mu1_model(x)
            predictions = np.array(mu1_hats - mu0_hats).squeeze()

        else:
            raise NotImplementedError('Base learner method not specified')

        return predictions


class SLearner:
    def __init__(self, method):
        self.name = "SLearner"
        self.method = method

        if self.method == 'rf':
            self.mux_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)

        elif self.method == 'lasso':
            self.mux_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.poly = PolynomialFeatures(degree=DEGREE_POLYNOMIALS, interaction_only=False, include_bias=False)

        elif self.method == 'nn':
            self.mux_model = clone_nn_regression(NN_SEQUENTIAL)

        else:
            raise NotImplementedError('Base learner method not or not correctly specified')

    def fit(self,
            x, y, w):

        x_w = np.concatenate((x, np.reshape(w, (len(w), 1))), axis=1)

        if self.method == 'rf':
            # 1: train mu_x
            self.mux_model.fit(x_w, y)

        elif self.method == 'lasso':
            # make polynomial features
            x_poly_train = self.poly.fit_transform(x_w)
            # 1: train mu_x
            self.mux_model.fit(x_poly_train, y)

        elif self.method == 'nn':
            # to tensor
            x_w = tf.convert_to_tensor(x_w)
            y = tf.convert_to_tensor(y)
            # 1: train mu_x
            self.mux_model.fit(x_w, y,
                               batch_size=BATCH_SIZE,
                               epochs=N_EPOCHS,
                               callbacks=CALLBACK,
                               validation_split=VALIDATION_SPLIT,
                               verbose=0
                               )
        else:
            raise NotImplementedError('Base learner method not specified in fit')

    def predict(self,
                x):

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
            # predictions
            predictions = mu1_hats - mu0_hats

        elif self.method == 'nn':
            # to tensor
            x_0 = tf.convert_to_tensor(x_0)
            x_1 = tf.convert_to_tensor(x_1)
            # 1: calculate hats of mu_x with X and W=1 or W=0
            mu0_hats = self.mux_model(x_0)
            mu1_hats = self.mux_model(x_1)
            # predictions
            predictions = np.array(mu1_hats - mu0_hats).squeeze()

        else:
            raise NotImplementedError('Base learner method not specified in predict')

        return predictions


class XLearner:  # TODO: comment what is what.
    def __init__(self, method):
        self.name = "XLearner"
        self.method = method

        if self.method == 'rf':
            self.mu0_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)
            self.mu1_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)
            self.ex_model = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)
            self.tau0_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                    random_state=RANDOM)
            self.tau1_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                    random_state=RANDOM)

        elif self.method == 'lasso':
            self.mu0_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM,
                                     max_iter=MAX_ITER)
            self.mu1_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.ex_model = LogisticRegressionCV(cv=KFold(K_FOLDS), penalty='l1', solver='saga', tol=TOLERANCE,
                                                 random_state=RANDOM,
                                                 max_iter=MAX_ITER)
            self.tau0_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.tau1_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.poly = PolynomialFeatures(degree=DEGREE_POLYNOMIALS, interaction_only=False, include_bias=False)

        elif self.method == 'nn':
            self.mu0_model = clone_nn_regression(NN_SEQUENTIAL)
            self.mu1_model = clone_nn_regression(NN_SEQUENTIAL)
            self.ex_model = clone_nn_classification(NN_SEQUENTIAL)
            self.tau0_model = clone_nn_regression(NN_SEQUENTIAL)
            self.tau1_model = clone_nn_regression(NN_SEQUENTIAL)

        else:
            raise NotImplementedError('Base learner method not or not correctly specified')

    def fit(self,
            x, y, w):
        if self.method == 'rf':
            # 1: train mu_0 and get imputed_1
            self.mu0_model.fit(x[w == 0], y[w == 0])
            imputed_1 = y[w == 1] - self.mu0_model.predict(x[w == 1])
            # 2: train mu_1 and get imputed_0
            self.mu1_model.fit(x[w == 1], y[w == 1])
            imputed_0 = self.mu1_model.predict(x[w == 0]) - y[w == 0]
            # 3: train tau_0
            self.tau0_model.fit(x[w == 0], imputed_0)
            # 4: train tau_1
            self.tau1_model.fit(x[w == 1], imputed_1)
            # 5: train e_x
            self.ex_model.fit(x, w)

        elif self.method == 'lasso':
            # make polynomial features
            x_poly_train = self.poly.fit_transform(x)
            # 1: train mu_0 and get imputed_1
            self.mu0_model.fit(x_poly_train[w == 0], y[w == 0])
            imputed_1 = y[w == 1] - self.mu0_model.predict(x_poly_train[w == 1])
            # 2: train mu_1 and get imputed_0
            self.mu1_model.fit(x_poly_train[w == 1], y[w == 1])
            imputed_0 = self.mu1_model.predict(x_poly_train[w == 0]) - y[w == 0]
            # 3: train tau_0
            self.tau0_model.fit(x_poly_train[w == 0], imputed_0)
            # 4: train tau_1
            self.tau1_model.fit(x_poly_train[w == 1], imputed_1)
            # 5: train e_x
            self.ex_model.fit(x_poly_train, w)

        elif self.method == 'nn':
            # to tensor
            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)
            w = tf.convert_to_tensor(w)
            # 1: train mu_0
            self.mu0_model.fit(x[w == 0], y[w == 0],
                               batch_size=BATCH_SIZE,
                               epochs=N_EPOCHS,
                               callbacks=CALLBACK,
                               validation_split=VALIDATION_SPLIT,
                               verbose=0
                               )
            imputed_1 = y[w == 1] - tf.squeeze(self.mu0_model(x[w == 1]))
            # 2: train mu_1
            self.mu1_model.fit(x[w == 1], y[w == 1],
                               batch_size=BATCH_SIZE,
                               epochs=N_EPOCHS,
                               callbacks=CALLBACK,
                               validation_split=VALIDATION_SPLIT,
                               verbose=0
                               )
            imputed_0 = tf.squeeze(self.mu1_model(x[w == 0])) - y[w == 0]
            # 3: train tau_0
            self.tau0_model.fit(x[w == 0], imputed_0,
                                batch_size=BATCH_SIZE,
                                epochs=N_EPOCHS,
                                callbacks=CALLBACK,
                                validation_split=VALIDATION_SPLIT,
                                verbose=0
                                )
            # 4: train tau_1
            self.tau1_model.fit(x[w == 1], imputed_1,
                                batch_size=BATCH_SIZE,
                                epochs=N_EPOCHS,
                                callbacks=CALLBACK,
                                validation_split=VALIDATION_SPLIT,
                                verbose=0
                                )
            # 5: train e_x
            self.ex_model.fit(x, w,
                              batch_size=BATCH_SIZE,
                              epochs=N_EPOCHS,
                              callbacks=CALLBACK,
                              validation_split=VALIDATION_SPLIT,
                              verbose=0
                              )

        else:
            raise NotImplementedError('Base learner method not specified')

    def predict(self,
                x):

        if self.method == 'rf':
            # 1: calculate hats of tau_0 and tau_1
            tau_0_hats = self.tau0_model.predict(x)
            tau_1_hats = self.tau1_model.predict(x)
            # 2: probabilities
            probs = self.ex_model.predict_proba(x)[:, 1]

        elif self.method == 'lasso':
            # make polynomial features
            x_poly_test = self.poly.fit_transform(x)
            # 1: calculate hats of tau_0 and tau_1
            tau_0_hats = self.tau0_model.predict(x_poly_test)
            tau_1_hats = self.tau1_model.predict(x_poly_test)
            # 2: probabilities
            probs = self.ex_model.predict_proba(x_poly_test)[:, 1]

        elif self.method == 'nn':
            # to tensor
            x = tf.convert_to_tensor(x)
            # 1: calculate hats of tau_0 and tau_1
            tau_0_hats = np.reshape(self.tau0_model(x), (len(x),))
            tau_1_hats = np.reshape(self.tau1_model(x), (len(x),))
            # 2: probabilities
            logit = self.ex_model(x)
            probs = np.array(keras.activations.sigmoid(logit)).squeeze()

        else:
            raise NotImplementedError('Base learner method not specified in predict')

        # 3: final predictions
        predictions = probs * tau_0_hats + (1 - probs) * tau_1_hats
        return predictions


class RLearner:
    def __init__(self, method):
        self.name = "RLearner"
        self.method = method

        if self.method == 'rf':
            self.mux_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)
            self.ex_model = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)
            self.tau_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)

        elif self.method == 'lasso':
            self.mux_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.ex_model = LogisticRegressionCV(cv=KFold(K_FOLDS), penalty='l1', solver='saga', tol=TOLERANCE,
                                                 random_state=RANDOM,
                                                 max_iter=MAX_ITER)
            self.tau_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.poly = PolynomialFeatures(degree=DEGREE_POLYNOMIALS, interaction_only=False, include_bias=False)

        elif self.method == 'nn':
            self.mux_model = clone_nn_regression(NN_SEQUENTIAL)
            self.ex_model = clone_nn_classification(NN_SEQUENTIAL)
            self.tau_model = clone_nn_regression(NN_SEQUENTIAL)

        else:
            raise NotImplementedError('Base learner method not or not correctly specified')

    def fit(self, x, y, w):

        if self.method == 'rf':
            # 1: fit mu_x
            self.mux_model.fit(x, y)
            # 2: fit ex
            self.ex_model.fit(x, w)
            # 3: calculate pseudo_outcomes & weights
            probs = self.ex_model.predict_proba(x)[:, 1]
            pseudo_outcomes = (y - self.mux_model.predict(x)) / (w - probs + EPSILON)
            weights = (w - probs) ** 2
            # 4: fit tau
            self.tau_model.fit(x, pseudo_outcomes, sample_weight=weights)

        elif self.method == 'lasso':
            x_poly_train = self.poly.fit_transform(x)
            # 1: fit mu_x
            self.mux_model.fit(x_poly_train, y)
            # 2: fit ex
            self.ex_model.fit(x_poly_train, w)
            # 3: calculate pseudo_outcomes & weights
            probs = self.ex_model.predict_proba(x_poly_train)[:, 1]
            pseudo_outcomes = (y - self.mux_model.predict(x_poly_train)) / (w - probs + EPSILON)
            weights = (w - probs) ** 2
            # 4: fit tau
            self.tau_model.fit(x_poly_train, pseudo_outcomes, sample_weight=weights)

        elif self.method == 'nn':
            # to tensor
            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)
            w = tf.convert_to_tensor(w)
            # 1: fit mu_x
            self.mux_model.fit(x, y,
                               batch_size=100,
                               epochs=N_EPOCHS,
                               callbacks=CALLBACK,
                               validation_split=0.3,
                               verbose=0
                               )
            # 2: fit ex
            self.ex_model.fit(x, w,
                              batch_size=100,
                              epochs=N_EPOCHS,
                              callbacks=CALLBACK,
                              validation_split=0.3,
                              verbose=0
                              )
            # 3: calculate pseudo_outcomes & weights
            probs = tf.squeeze(keras.activations.sigmoid(self.ex_model(x)))
            pseudo_outcomes = (y - tf.squeeze(self.mux_model(x))) / (w - probs + EPSILON)
            weights = (w - probs) ** 2
            # 4: fit tau
            self.tau_model.fit(x, pseudo_outcomes,
                               sample_weight=weights,
                               batch_size=BATCH_SIZE,
                               epochs=N_EPOCHS,
                               callbacks=CALLBACK,
                               validation_split=VALIDATION_SPLIT,
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
            # to tensor
            x = tf.convert_to_tensor(x)
            # predict
            predictions = np.array(self.tau_model(x)).squeeze()

        else:
            raise NotImplementedError('Base learner method not specified in predict')

        return predictions


class DRLearner:
    def __init__(self, method):
        self.name = "DRLearner"
        self.method = method

        if self.method == 'rf':
            self.mu0_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)
            self.mu1_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)
            self.ex_model = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)
            self.tau_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)

        elif self.method == 'lasso':
            self.mu0_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.mu1_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.ex_model = LogisticRegressionCV(cv=KFold(K_FOLDS), penalty='l1', solver='saga', tol=TOLERANCE,
                                                 random_state=RANDOM,
                                                 max_iter=MAX_ITER)
            self.tau_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.poly = PolynomialFeatures(degree=DEGREE_POLYNOMIALS, interaction_only=False, include_bias=False)

        elif self.method == 'nn':
            self.mu0_model = clone_nn_regression(NN_SEQUENTIAL)
            self.mu1_model = clone_nn_regression(NN_SEQUENTIAL)
            self.ex_model = clone_nn_classification(NN_SEQUENTIAL)
            self.tau_model = clone_nn_regression(NN_SEQUENTIAL)

        else:
            raise NotImplementedError('Base learner method not or not correctly specified')

    def fit(self, x, y, w):

        if self.method == 'rf':
            # 1: fit mu_0
            self.mu0_model.fit(x[w == 0], y[w == 0])
            # 2: fit mu_1
            self.mu1_model.fit(x[w == 1], y[w == 1])
            # 3: fit ex
            self.ex_model.fit(x, w)
            probs = self.ex_model.predict_proba(x)[:, 1]
            # calculate pseudo_outcomes
            mu_w = w * self.mu1_model.predict(x) + (1 - w) * self.mu0_model.predict(x)
            pseudo_outcomes = (w - probs) / (probs * (1 - probs) + EPSILON) * (y - mu_w) + self.mu1_model.predict(
                x) - self.mu0_model.predict(x)
            # 4 fit tau
            self.tau_model.fit(x, pseudo_outcomes)

        elif self.method == 'lasso':
            # poly
            x_poly_train = self.poly.fit_transform(x)
            # 1: fit mu_0
            self.mu0_model.fit(x_poly_train[w == 0], y[w == 0])
            # 2: fit mu_1
            self.mu1_model.fit(x_poly_train[w == 1], y[w == 1])
            # 3: fit ex
            self.ex_model.fit(x_poly_train, w)
            probs = self.ex_model.predict_proba(x_poly_train)[:, 1]
            # calculate pseudo_outcomes
            mu_w = w * self.mu1_model.predict(x_poly_train) + (1 - w) * self.mu0_model.predict(x_poly_train)
            pseudo_outcomes = (w - probs) / (probs * (1 - probs) + EPSILON) * (y - mu_w) + self.mu1_model.predict(
                x_poly_train) - self.mu0_model.predict(x_poly_train)
            # 4 fit tau
            self.tau_model.fit(x_poly_train, pseudo_outcomes)

        elif self.method == 'nn':
            # to tensor
            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)
            w = tf.convert_to_tensor(w)
            # 1: fit mu_0
            self.mu0_model.fit(x[w == 0], y[w == 0],
                               batch_size=BATCH_SIZE,
                               epochs=N_EPOCHS,
                               callbacks=CALLBACK,
                               validation_split=VALIDATION_SPLIT,
                               verbose=0
                               )
            # 2: fit mu_1
            self.mu1_model.fit(x[w == 1], y[w == 1],
                               batch_size=BATCH_SIZE,
                               epochs=N_EPOCHS,
                               callbacks=CALLBACK,
                               validation_split=VALIDATION_SPLIT,
                               verbose=0
                               )
            # 3: fit ex
            self.ex_model.fit(x, w,
                              batch_size=BATCH_SIZE,
                              epochs=N_EPOCHS,
                              callbacks=CALLBACK,
                              validation_split=VALIDATION_SPLIT,
                              verbose=0
                              )
            probs = tf.squeeze(keras.activations.sigmoid(self.ex_model(x)))
            # calculate pseudo_outcomes
            mu_0_hats = self.mu0_model(x)
            mu_1_hats = self.mu1_model(x)
            mu_w = w * mu_1_hats + (1 - w) * mu_0_hats
            pseudo_outcomes = (w - probs) / (probs * (1 - probs) + EPSILON) * (y - mu_w) + mu_1_hats - mu_0_hats
            # 4 fit tau
            self.tau_model.fit(x, pseudo_outcomes,
                               batch_size=BATCH_SIZE,
                               epochs=N_EPOCHS,
                               callbacks=CALLBACK,
                               validation_split=VALIDATION_SPLIT,
                               verbose=0
                               )

    def predict(self, x):

        if self.method == 'rf':
            predictions = self.tau_model.predict(x)

        elif self.method == 'lasso':
            x_poly_test = self.poly.fit_transform(x)
            predictions = self.tau_model.predict(x_poly_test)

        elif self.method == 'nn':
            # to tensor
            x = tf.convert_to_tensor(x)
            # predict
            predictions = np.array(self.tau_model(x)).squeeze()

        else:
            raise NotImplementedError('Base learner method not specified in predict')

        return predictions


class RALearner:
    def __init__(self, method):
        self.name = "RALearner"
        self.method = method

        if self.method == 'rf':
            self.mu0_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)
            self.mu1_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)
            self.tau_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)

        elif self.method == 'lasso':
            self.mu0_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.mu1_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.tau_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.poly = PolynomialFeatures(degree=DEGREE_POLYNOMIALS, interaction_only=False, include_bias=False)

        elif self.method == 'nn':
            self.mu0_model = clone_nn_regression(NN_SEQUENTIAL)
            self.mu1_model = clone_nn_regression(NN_SEQUENTIAL)
            self.tau_model = clone_nn_regression(NN_SEQUENTIAL)

        else:
            raise NotImplementedError('Base learner method not or not correctly specified')

    def fit(self, x, y, w):
        if self.method == 'rf':
            # 1: fit mu_0
            self.mu0_model.fit(x[w == 0], y[w == 0])
            # 2: fit mu_1
            self.mu1_model.fit(x[w == 1], y[w == 1])
            # calculate pseudo_outcomes
            pseudo_outcomes = w * (y - self.mu0_model.predict(x)) + (1 - w) * (self.mu1_model.predict(x) - y)
            # 4 fit tau
            self.tau_model.fit(x, pseudo_outcomes)

        elif self.method == 'lasso':
            # poly
            x_poly_train = self.poly.fit_transform(x)
            # 1: fit mu_0
            self.mu0_model.fit(x_poly_train[w == 0], y[w == 0])
            # 2: fit mu_1
            self.mu1_model.fit(x_poly_train[w == 1], y[w == 1])
            # calculate pseudo_outcomes
            pseudo_outcomes = w * (y - self.mu0_model.predict(x_poly_train)) + (1 - w) * (
                    self.mu1_model.predict(x_poly_train) - y)
            # 4 fit tau
            self.tau_model.fit(x_poly_train, pseudo_outcomes)

        elif self.method == 'nn':
            # to tensor
            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)
            w = tf.convert_to_tensor(w)
            # 1: fit mu_0
            self.mu0_model.fit(x[w == 0], y[w == 0],
                               batch_size=BATCH_SIZE,
                               epochs=N_EPOCHS,
                               callbacks=CALLBACK,
                               validation_split=VALIDATION_SPLIT,
                               verbose=0
                               )
            # 2: fit mu_1
            self.mu1_model.fit(x[w == 1], y[w == 1],
                               batch_size=BATCH_SIZE,
                               epochs=N_EPOCHS,
                               callbacks=CALLBACK,
                               validation_split=VALIDATION_SPLIT,
                               verbose=0
                               )
            # calculate pseudo_outcomes
            mu0_predictions = tf.squeeze(self.mu0_model(x))  # TODO: do it like that for the other metalearner
            mu1_predictions = tf.squeeze(self.mu1_model(x))
            pseudo_outcomes = w * (y - mu0_predictions) + (1 - w) * (mu1_predictions - y)
            # 4 fit tau
            self.tau_model.fit(x, pseudo_outcomes,
                               batch_size=BATCH_SIZE,
                               epochs=N_EPOCHS,
                               callbacks=CALLBACK,
                               validation_split=VALIDATION_SPLIT,
                               verbose=0
                               )

    def predict(self, x):

        if self.method == 'rf':
            predictions = self.tau_model.predict(x)

        elif self.method == 'lasso':
            x_poly_test = self.poly.fit_transform(x)
            predictions = self.tau_model.predict(x_poly_test)

        elif self.method == 'nn':
            # to tensor
            x = tf.convert_to_tensor(x)
            # predict
            predictions = np.array(self.tau_model(x)).squeeze()

        else:
            raise NotImplementedError('Base learner method not specified')

        return predictions


class PWLearner:
    def __init__(self, method):
        self.name = "PWLearner"
        self.method = method

        if self.method == 'rf':
            self.ex_model = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)
            self.tau_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)

        elif self.method == 'lasso':
            self.ex_model = LogisticRegressionCV(cv=KFold(K_FOLDS), penalty='l1', solver='saga', tol=TOLERANCE,
                                                 random_state=RANDOM,
                                                 max_iter=MAX_ITER)
            self.tau_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.poly = PolynomialFeatures(degree=DEGREE_POLYNOMIALS, interaction_only=False, include_bias=False)

        elif self.method == 'nn':
            self.ex_model = clone_nn_classification(NN_SEQUENTIAL)
            self.tau_model = clone_nn_regression(NN_SEQUENTIAL)

        else:
            raise NotImplementedError('Base learner method not or not correctly specified')

    def fit(self, x, y, w):

        if self.method == 'rf':
            # 3: fit ex
            self.ex_model.fit(x, w)
            probs = self.ex_model.predict_proba(x)[:, 1]
            # calculate pseudo_outcomes
            pseudo_outcomes = (w / (probs + EPSILON) - (1 - w) / (1 - probs + EPSILON)) * y
            # 4 fit tau
            self.tau_model.fit(x, pseudo_outcomes)

        elif self.method == 'lasso':
            # poly
            x_poly_train = self.poly.fit_transform(x)
            # 3: fit ex
            self.ex_model.fit(x_poly_train, w)
            probs = self.ex_model.predict_proba(x_poly_train)[:, 1]
            # calculate pseudo_outcomes
            pseudo_outcomes = (w / (probs + EPSILON) - (1 - w) / (1 - probs + EPSILON)) * y
            # 4 fit tau
            self.tau_model.fit(x_poly_train, pseudo_outcomes)

        elif self.method == 'nn':
            # to tensor
            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)
            w = tf.convert_to_tensor(w)
            # 3: fit ex
            self.ex_model.fit(x, w,
                              batch_size=BATCH_SIZE,
                              epochs=N_EPOCHS,
                              callbacks=CALLBACK,
                              validation_split=VALIDATION_SPLIT,
                              verbose=0
                              )
            probs = tf.squeeze(keras.activations.sigmoid(self.ex_model(x)))
            # calculate pseudo_outcomes
            pseudo_outcomes = (w / (probs + EPSILON) - (1 - w) / (1 - probs + EPSILON)) * y
            # 4 fit tau
            self.tau_model.fit(x, pseudo_outcomes,
                               batch_size=BATCH_SIZE,
                               epochs=N_EPOCHS,
                               callbacks=CALLBACK,
                               validation_split=VALIDATION_SPLIT,
                               verbose=0
                               )

    def predict(self, x):

        if self.method == 'rf':
            predictions = self.tau_model.predict(x)

        elif self.method == 'lasso':
            x_poly_test = self.poly.fit_transform(x)
            predictions = self.tau_model.predict(x_poly_test)

        elif self.method == 'nn':
            # to tensor
            x = tf.convert_to_tensor(x)
            # predict
            predictions = np.array(self.tau_model(x)).squeeze()

        else:
            raise NotImplementedError('Base learner method not specified in predict')

        return predictions


class ULearner:
    def __init__(self, method):
        self.name = "ULearner"
        self.method = method

        if self.method == 'rf':
            self.mux_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)
            self.ex_model = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)
            self.tau_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)

        elif self.method == 'lasso':
            self.mux_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.ex_model = LogisticRegressionCV(cv=KFold(K_FOLDS), penalty='l1', solver='saga', tol=TOLERANCE,
                                                 random_state=RANDOM,
                                                 max_iter=MAX_ITER)
            self.tau_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.poly = PolynomialFeatures(degree=DEGREE_POLYNOMIALS, interaction_only=False, include_bias=False)

        elif self.method == 'nn':
            self.mux_model = clone_nn_regression(NN_SEQUENTIAL)
            self.ex_model = clone_nn_classification(NN_SEQUENTIAL)
            self.tau_model = clone_nn_regression(NN_SEQUENTIAL)

        else:
            raise NotImplementedError('Base learner method not or not correctly specified')

    def fit(self, x, y, w):

        if self.method == 'rf':
            # 2: fit mu_x
            self.mux_model.fit(x, y)
            # 3: fit ex
            self.ex_model.fit(x, w)
            probs = self.ex_model.predict_proba(x)[:, 1]
            # calculate residuals
            residuals = (y - self.mux_model.predict(x)) / (w - probs + EPSILON)
            # 4 fit tau
            self.tau_model.fit(x, residuals)

        elif self.method == 'lasso':
            x_poly_train = self.poly.fit_transform(x)
            # 2: fit mu_x
            self.mux_model.fit(x_poly_train, y)
            # 3: fit ex
            self.ex_model.fit(x_poly_train, w)
            probs = self.ex_model.predict_proba(x_poly_train)[:, 1]
            # calculate pseudo_outcomes
            residuals = (y - self.mux_model.predict(x_poly_train)) / (w - probs + EPSILON)
            # 4 fit tau
            self.tau_model.fit(x_poly_train, residuals)

        elif self.method == 'nn':
            # to tensor
            x = tf.convert_to_tensor(x)
            y = tf.convert_to_tensor(y)
            w = tf.convert_to_tensor(w)
            # 1: fit mu_x
            self.mux_model.fit(x, y,
                               batch_size=BATCH_SIZE,
                               epochs=N_EPOCHS,
                               callbacks=CALLBACK,
                               validation_split=VALIDATION_SPLIT,
                               verbose=0
                               )
            # 3: fit ex
            self.ex_model.fit(x, w,
                              batch_size=BATCH_SIZE,
                              epochs=N_EPOCHS,
                              callbacks=CALLBACK,
                              validation_split=VALIDATION_SPLIT,
                              verbose=0
                              )
            probs = tf.squeeze(keras.activations.sigmoid(self.ex_model(x)))
            # calculate pseudo_outcomes
            mu_x_predictions = tf.squeeze(self.mux_model(x))
            residuals = (y - mu_x_predictions) / (w - probs + EPSILON)
            # 4 fit tau
            self.tau_model.fit(x, residuals,
                               batch_size=BATCH_SIZE,
                               epochs=N_EPOCHS,
                               callbacks=CALLBACK,
                               validation_split=VALIDATION_SPLIT,
                               verbose=0
                               )

    def predict(self, x):

        if self.method == 'rf':
            predictions = self.tau_model.predict(x)

        elif self.method == 'lasso':
            x_poly_test = self.poly.fit_transform(x)
            predictions = self.tau_model.predict(x_poly_test)

        elif self.method == 'nn':
            # to tensor
            x = tf.convert_to_tensor(x)
            # predict
            predictions = np.array(self.tau_model(x)).squeeze()

        else:
            raise NotImplementedError('Base learner method not specified in predict')

        return predictions
