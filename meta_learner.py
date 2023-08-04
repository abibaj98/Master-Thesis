# import packages
import tensorflow as tf
import keras.activations
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
# import from files
from default_parameters import *
from neural_networks import clone_nn_regression, clone_nn_classification, NN_SEQUENTIAL, CALLBACK


class TLearner:  # TODO: comment what is what.
    def __init__(self, method):
        self.method = method
        self.name = f"TLearner, {self.method}"

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
            raise ValueError('Base learner method not (or not correctly) specified. Can be "rf", "lasso" or "nn".')

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
            raise ValueError('Base learner method not (or not correctly) specified. Can be "rf", "lasso" or "nn".')

        return predictions


class SLearner:
    def __init__(self, method):
        self.method = method
        self.name = f"SLearner, {self.method}"

        if self.method == 'rf':
            self.mux_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)

        elif self.method == 'lasso':
            self.mux_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.poly = PolynomialFeatures(degree=DEGREE_POLYNOMIALS, interaction_only=False, include_bias=False)

        elif self.method == 'nn':
            self.mux_model = clone_nn_regression(NN_SEQUENTIAL)

        else:
            raise ValueError('Base learner method not (or not correctly) specified. Can be "rf", "lasso" or "nn".')

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
            raise ValueError('Base learner method not (or not correctly) specified. Can be "rf", "lasso" or "nn".')

        return predictions


class XLearner:  # TODO: comment what is what.
    def __init__(self, method):
        self.method = method
        self.name = f"XLearner, {self.method}"

        if self.method == 'rf':
            self.ex_model = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=RANDOM)
            self.tau0_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=RANDOM)
            self.tau1_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=RANDOM)

        elif self.method == 'lasso':
            self.ex_model = LogisticRegressionCV(cv=KFold(K_FOLDS), penalty='l1', solver='saga', tol=TOLERANCE,
                                                 random_state=RANDOM,
                                                 max_iter=MAX_ITER)
            self.tau0_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.tau1_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.poly = PolynomialFeatures(degree=DEGREE_POLYNOMIALS, interaction_only=False, include_bias=False)

        elif self.method == 'nn':
            self.ex_model = clone_nn_classification(NN_SEQUENTIAL)
            self.tau0_model = clone_nn_regression(NN_SEQUENTIAL)
            self.tau1_model = clone_nn_regression(NN_SEQUENTIAL)

        else:
            raise ValueError('Base learner method not (or not correctly) specified. Can be "rf", "lasso" or "nn".')

    @staticmethod
    def compute_hats_rf(x_fit, y_fit, w_fit, x_pred):
        # set models
        temp_mu0 = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=RANDOM)
        temp_mu1 = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=RANDOM)
        # fit
        temp_mu0.fit(x_fit[w_fit == 0], y_fit[w_fit == 0])
        temp_mu1.fit(x_fit[w_fit == 1], y_fit[w_fit == 1])
        # hats
        mu0_hat = temp_mu0.predict(x_pred)
        mu1_hat = temp_mu1.predict(x_pred)
        return mu0_hat, mu1_hat

    def compute_hats_lasso(self, x_fit, y_fit, w_fit, x_pred):
        # poly transformation
        x_poly_fit = self.poly.fit_transform(x_fit)
        x_poly_pred = self.poly.fit_transform(x_pred)
        # set models
        temp_mu0 = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
        temp_mu1 = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
        # fit
        temp_mu0.fit(x_poly_fit[w_fit == 0], y_fit[w_fit == 0])
        temp_mu1.fit(x_poly_fit[w_fit == 1], y_fit[w_fit == 1])
        # hats
        mu0_hat = temp_mu0.predict(x_poly_pred)
        mu1_hat = temp_mu1.predict(x_poly_pred)
        return mu0_hat, mu1_hat

    @staticmethod
    def compute_hats_nn(x_fit, y_fit, w_fit, x_pred):
        # to tensor
        x_fit = tf.convert_to_tensor(x_fit)
        y_fit = tf.convert_to_tensor(y_fit)
        w_fit = tf.convert_to_tensor(w_fit)
        x_pred = tf.convert_to_tensor(x_pred)
        # set models
        temp_mu0 = clone_nn_regression(NN_SEQUENTIAL)
        temp_mu1 = clone_nn_regression(NN_SEQUENTIAL)
        # fit
        temp_mu0.fit(x_fit[w_fit == 0], y_fit[w_fit == 0],
                     batch_size=BATCH_SIZE,
                     epochs=N_EPOCHS,
                     callbacks=CALLBACK,
                     validation_split=VALIDATION_SPLIT,
                     verbose=0)
        temp_mu1.fit(x_fit[w_fit == 1], y_fit[w_fit == 1],
                     batch_size=BATCH_SIZE,
                     epochs=N_EPOCHS,
                     callbacks=CALLBACK,
                     validation_split=VALIDATION_SPLIT,
                     verbose=0)
        # hats
        mu0_hat = tf.squeeze(temp_mu0(x_pred))
        mu1_hat = tf.squeeze(temp_mu1(x_pred))
        return mu0_hat, mu1_hat

    def fit(self,
            x, y, w):

        if self.method == 'rf':
            # hats
            mu0_hat, mu1_hat = self.compute_hats_rf(x, y, w, x)
            # imputed outcomes
            imputed_0 = mu1_hat[w == 0] - y[w == 0]
            imputed_1 = y[w == 1] - mu0_hat[w == 1]
            # 3: fit ex, tau0 and tau1
            self.ex_model.fit(x, w)
            self.tau0_model.fit(x[w == 0], imputed_0)
            self.tau1_model.fit(x[w == 1], imputed_1)

        elif self.method == 'lasso':
            # hats
            mu0_hat, mu1_hat = self.compute_hats_lasso(x, y, w, x)
            # imputed outcomes
            imputed_0 = mu1_hat[w == 0] - y[w == 0]
            imputed_1 = y[w == 1] - mu0_hat[w == 1]
            # to poly
            x_poly = self.poly.fit_transform(x)
            # 3: fit tau0 and tau1
            self.ex_model.fit(x_poly, w)
            self.tau0_model.fit(x_poly[w == 0], imputed_0)
            self.tau1_model.fit(x_poly[w == 1], imputed_1)

        elif self.method == 'nn':
            # hats
            mu0_hat, mu1_hat = self.compute_hats_nn(x, y, w, x)
            # imputed outcomes
            imputed_0 = mu1_hat[w == 0] - y[w == 0]
            imputed_1 = y[w == 1] - mu0_hat[w == 1]
            # 3: fit ex, tau0 and tau1
            self.ex_model.fit(x, w,
                              batch_size=BATCH_SIZE,
                              epochs=N_EPOCHS,
                              callbacks=CALLBACK,
                              validation_split=VALIDATION_SPLIT,
                              verbose=0
                              )
            self.tau0_model.fit(x[w == 0], imputed_0,
                                batch_size=BATCH_SIZE,
                                epochs=N_EPOCHS,
                                callbacks=CALLBACK,
                                validation_split=VALIDATION_SPLIT,
                                verbose=0
                                )
            self.tau1_model.fit(x[w == 1], imputed_1,
                                batch_size=BATCH_SIZE,
                                epochs=N_EPOCHS,
                                callbacks=CALLBACK,
                                validation_split=VALIDATION_SPLIT,
                                verbose=0
                                )

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
            tau_0_hats = np.array(self.tau0_model(x)).squeeze()
            tau_1_hats = np.array(self.tau1_model(x)).squeeze()
            # 2: probabilities
            logit = self.ex_model(x)
            probs = np.array(keras.activations.sigmoid(logit)).squeeze()

        else:
            raise ValueError('Base learner method not (or not correctly) specified. Can be "rf", "lasso" or "nn".')

        # 3: final predictions
        predictions = probs * tau_0_hats + (1 - probs) * tau_1_hats
        return predictions


class RLearner:
    def __init__(self, method):
        self.method = method
        self.name = f"RLearner, {self.method}"

        if self.method == 'rf':
            self.tau_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)

        elif self.method == 'lasso':
            self.tau_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.poly = PolynomialFeatures(degree=DEGREE_POLYNOMIALS, interaction_only=False, include_bias=False)

        elif self.method == 'nn':
            self.tau_model = clone_nn_regression(NN_SEQUENTIAL)

        else:
            raise ValueError('Base learner method not (or not correctly) specified. Can be "rf", "lasso" or "nn".')

    @staticmethod
    def compute_hats_rf(x_fit, y_fit, w_fit, x_pred):
        # set models
        temp_mux = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=RANDOM)
        temp_ex = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=RANDOM)
        # fit
        temp_mux.fit(x_fit, y_fit)
        temp_ex.fit(x_fit, w_fit)
        # hats
        mux_hat = temp_mux.predict(x_pred)
        probs = temp_ex.predict_proba(x_pred)[:, 1]
        # clipping probabilities
        np.clip(probs, MIN_CLIP, MAX_CLIP, out=probs)
        return mux_hat, probs

    def compute_hats_lasso(self, x_fit, y_fit, w_fit, x_pred):
        # poly transformation
        x_poly_fit = self.poly.fit_transform(x_fit)
        x_poly_pred = self.poly.fit_transform(x_pred)
        # set models
        temp_mux = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
        temp_ex = LogisticRegressionCV(cv=KFold(K_FOLDS), penalty='l1', solver='saga', tol=TOLERANCE,
                                       random_state=RANDOM,
                                       max_iter=MAX_ITER)
        # fit
        temp_mux.fit(x_poly_fit, y_fit)
        temp_ex.fit(x_poly_fit, w_fit)
        # hats
        mux_hat = temp_mux.predict(x_poly_pred)
        probs = temp_ex.predict_proba(x_poly_pred)[:, 1]
        # clipping probabilities
        np.clip(probs, MIN_CLIP, MAX_CLIP, out=probs)
        return mux_hat, probs

    @staticmethod
    def compute_hats_nn(x_fit, y_fit, w_fit, x_pred):
        # to tensor
        x_fit = tf.convert_to_tensor(x_fit)
        y_fit = tf.convert_to_tensor(y_fit)
        w_fit = tf.convert_to_tensor(w_fit)
        x_pred = tf.convert_to_tensor(x_pred)
        # set models
        temp_mux = clone_nn_regression(NN_SEQUENTIAL)
        temp_ex = clone_nn_classification(NN_SEQUENTIAL)
        # fit
        temp_mux.fit(x_fit, y_fit,
                     batch_size=BATCH_SIZE,
                     epochs=N_EPOCHS,
                     callbacks=CALLBACK,
                     validation_split=VALIDATION_SPLIT,
                     verbose=0)
        temp_ex.fit(x_fit, w_fit,
                    batch_size=BATCH_SIZE,
                    epochs=N_EPOCHS,
                    callbacks=CALLBACK,
                    validation_split=VALIDATION_SPLIT,
                    verbose=0)
        # hats
        mux_hat = tf.squeeze(temp_mux(x_pred))
        probs = tf.squeeze(keras.activations.sigmoid(temp_ex(x_pred)))
        # clipping probabilities
        probs = tf.clip_by_value(probs, MIN_CLIP, MAX_CLIP)
        return mux_hat, probs

    def fit(self,
            x, y, w):
        if self.method == 'rf':
            # hats
            mux_hat, probs = self.compute_hats_rf(x, y, w, x)
            # pseudo-outcomes
            pseudo_outcomes = (y - mux_hat) / (w - probs)
            weights = (w - probs) ** 2
            # 3: fit tau
            self.tau_model.fit(x, pseudo_outcomes, sample_weight=weights)

        elif self.method == 'lasso':
            # hats
            mux_hat, probs = self.compute_hats_lasso(x, y, w, x)
            # pseudo-outcomes
            pseudo_outcomes = (y - mux_hat) / (w - probs)
            weights = (w - probs) ** 2
            # to poly
            x_poly = self.poly.fit_transform(x)
            # 3: fit tau
            self.tau_model.fit(x_poly, pseudo_outcomes, sample_weight=weights)

        elif self.method == 'nn':
            # hats
            mux_hat, probs = self.compute_hats_nn(x, y, w, x)
            # pseudo-outcomes
            pseudo_outcomes = (y - mux_hat) / (w - probs)
            weights = (w - probs) ** 2
            # 3: fit tau
            self.tau_model.fit(x, pseudo_outcomes,
                               sample_weight=weights,
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
            raise ValueError('Base learner method not (or not correctly) specified. Can be "rf", "lasso" or "nn".')

        return predictions


class DRLearner:
    def __init__(self, method):
        self.method = method
        self.name = f"DRLearner, {self.method}"

        if self.method == 'rf':
            self.ex_model = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)
            self.tau_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)

        elif self.method == 'lasso':
            self.tau_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.poly = PolynomialFeatures(degree=DEGREE_POLYNOMIALS, interaction_only=False, include_bias=False)

        elif self.method == 'nn':
            self.tau_model = clone_nn_regression(NN_SEQUENTIAL)

        else:
            raise ValueError('Base learner method not (or not correctly) specified. Can be "rf", "lasso" or "nn".')

    @staticmethod
    def compute_hats_rf(x_fit, y_fit, w_fit, x_pred):
        # set models
        temp_mu0 = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=RANDOM)
        temp_mu1 = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=RANDOM)
        temp_ex = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=RANDOM)
        # fit
        temp_mu0.fit(x_fit[w_fit == 0], y_fit[w_fit == 0])
        temp_mu1.fit(x_fit[w_fit == 1], y_fit[w_fit == 1])
        temp_ex.fit(x_fit, w_fit)
        # hats
        mu0_hat = temp_mu0.predict(x_pred)
        mu1_hat = temp_mu1.predict(x_pred)
        probs = temp_ex.predict_proba(x_pred)[:, 1]
        # clipping probabilities
        np.clip(probs, MIN_CLIP, MAX_CLIP, out=probs)
        return mu0_hat, mu1_hat, probs

    def compute_hats_lasso(self, x_fit, y_fit, w_fit, x_pred):
        # poly transformation
        x_poly_fit = self.poly.fit_transform(x_fit)
        x_poly_pred = self.poly.fit_transform(x_pred)
        # set models
        temp_mu0 = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
        temp_mu1 = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
        temp_ex = LogisticRegressionCV(cv=KFold(K_FOLDS), penalty='l1', solver='saga', tol=TOLERANCE,
                                       random_state=RANDOM,
                                       max_iter=MAX_ITER)
        # fit
        temp_mu0.fit(x_poly_fit[w_fit == 0], y_fit[w_fit == 0])
        temp_mu1.fit(x_poly_fit[w_fit == 1], y_fit[w_fit == 1])
        temp_ex.fit(x_poly_fit, w_fit)
        # hats
        mu0_hat = temp_mu0.predict(x_poly_pred)
        mu1_hat = temp_mu1.predict(x_poly_pred)
        probs = temp_ex.predict_proba(x_poly_pred)[:, 1]
        # clipping probabilities
        np.clip(probs, MIN_CLIP, MAX_CLIP, out=probs)
        return mu0_hat, mu1_hat, probs

    @staticmethod
    def compute_hats_nn(x_fit, y_fit, w_fit, x_pred):
        # to tensor
        x_fit = tf.convert_to_tensor(x_fit)
        y_fit = tf.convert_to_tensor(y_fit)
        w_fit = tf.convert_to_tensor(w_fit)
        x_pred = tf.convert_to_tensor(x_pred)
        # set models
        temp_mu0 = clone_nn_regression(NN_SEQUENTIAL)
        temp_mu1 = clone_nn_regression(NN_SEQUENTIAL)
        temp_ex = clone_nn_classification(NN_SEQUENTIAL)
        # fit
        temp_mu0.fit(x_fit[w_fit == 0], y_fit[w_fit == 0],
                     batch_size=BATCH_SIZE,
                     epochs=N_EPOCHS,
                     callbacks=CALLBACK,
                     validation_split=VALIDATION_SPLIT,
                     verbose=0)
        temp_mu1.fit(x_fit[w_fit == 1], y_fit[w_fit == 1],
                     batch_size=BATCH_SIZE,
                     epochs=N_EPOCHS,
                     callbacks=CALLBACK,
                     validation_split=VALIDATION_SPLIT,
                     verbose=0)
        temp_ex.fit(x_fit, w_fit,
                    batch_size=BATCH_SIZE,
                    epochs=N_EPOCHS,
                    callbacks=CALLBACK,
                    validation_split=VALIDATION_SPLIT,
                    verbose=0)
        # hats
        mu0_hat = tf.squeeze(temp_mu0(x_pred))
        mu1_hat = tf.squeeze(temp_mu1(x_pred))
        probs = tf.squeeze(keras.activations.sigmoid(temp_ex(x_pred)))
        # clipping probabilities
        probs = tf.clip_by_value(probs, MIN_CLIP, MAX_CLIP)
        return mu0_hat, mu1_hat, probs

    def fit(self,
            x, y, w):
        if self.method == 'rf':
            # hats
            mu0_hat, mu1_hat, probs = self.compute_hats_rf(x, y, w, x)
            # pseudo-outcomes
            mu_w = w * mu1_hat + (1 - w) * mu0_hat
            pseudo_outcomes = (w - probs) / (probs * (1 - probs)) * (y - mu_w) + mu1_hat - mu0_hat
            # 3: fit tau
            self.tau_model.fit(x, pseudo_outcomes)

        elif self.method == 'lasso':
            # hats
            mu0_hat, mu1_hat, probs = self.compute_hats_lasso(x, y, w, x)
            # pseudo-outcomes
            mu_w = w * mu1_hat + (1 - w) * mu0_hat
            pseudo_outcomes = (w - probs) / (probs * (1 - probs)) * (y - mu_w) + mu1_hat - mu0_hat
            # to poly
            x_poly = self.poly.fit_transform(x)
            # 3: fit tau
            self.tau_model.fit(x_poly, pseudo_outcomes)

        elif self.method == 'nn':
            # hats
            mu0_hat, mu1_hat, probs = self.compute_hats_nn(x, y, w, x)
            # pseudo-outcomes
            mu_w = w * mu1_hat + (1 - w) * mu0_hat
            pseudo_outcomes = (w - probs) / (probs * (1 - probs)) * (y - mu_w) + mu1_hat - mu0_hat
            # 3: fit tau
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
            raise ValueError('Base learner method not (or not correctly) specified. Can be "rf", "lasso" or "nn".')

        return predictions


class RALearner:
    def __init__(self, method):
        self.method = method
        self.name = f"RALearner, {self.method}"

        if self.method == 'rf':
            self.tau_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)

        elif self.method == 'lasso':
            self.tau_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.poly = PolynomialFeatures(degree=DEGREE_POLYNOMIALS, interaction_only=False, include_bias=False)

        elif self.method == 'nn':
            self.tau_model = clone_nn_regression(NN_SEQUENTIAL)

        else:
            raise ValueError('Base learner method not (or not correctly) specified. Can be "rf", "lasso" or "nn".')

    @staticmethod
    def compute_hats_rf(x_fit, y_fit, w_fit, x_pred):
        # set models
        temp_mu0 = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=RANDOM)
        temp_mu1 = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=RANDOM)
        # fit
        temp_mu0.fit(x_fit[w_fit == 0], y_fit[w_fit == 0])
        temp_mu1.fit(x_fit[w_fit == 1], y_fit[w_fit == 1])
        # hats
        mu0_hat = temp_mu0.predict(x_pred)
        mu1_hat = temp_mu1.predict(x_pred)
        return mu0_hat, mu1_hat

    def compute_hats_lasso(self, x_fit, y_fit, w_fit, x_pred):
        # poly transformation
        x_poly_fit = self.poly.fit_transform(x_fit)
        x_poly_pred = self.poly.fit_transform(x_pred)
        # set models
        temp_mu0 = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
        temp_mu1 = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
        # fit
        temp_mu0.fit(x_poly_fit[w_fit == 0], y_fit[w_fit == 0])
        temp_mu1.fit(x_poly_fit[w_fit == 1], y_fit[w_fit == 1])
        # hats
        mu0_hat = temp_mu0.predict(x_poly_pred)
        mu1_hat = temp_mu1.predict(x_poly_pred)
        return mu0_hat, mu1_hat

    @staticmethod
    def compute_hats_nn(x_fit, y_fit, w_fit, x_pred):
        # to tensor
        x_fit = tf.convert_to_tensor(x_fit)
        y_fit = tf.convert_to_tensor(y_fit)
        w_fit = tf.convert_to_tensor(w_fit)
        x_pred = tf.convert_to_tensor(x_pred)
        # set models
        temp_mu0 = clone_nn_regression(NN_SEQUENTIAL)
        temp_mu1 = clone_nn_regression(NN_SEQUENTIAL)
        # fit
        temp_mu0.fit(x_fit[w_fit == 0], y_fit[w_fit == 0],
                     batch_size=BATCH_SIZE,
                     epochs=N_EPOCHS,
                     callbacks=CALLBACK,
                     validation_split=VALIDATION_SPLIT,
                     verbose=0)
        temp_mu1.fit(x_fit[w_fit == 1], y_fit[w_fit == 1],
                     batch_size=BATCH_SIZE,
                     epochs=N_EPOCHS,
                     callbacks=CALLBACK,
                     validation_split=VALIDATION_SPLIT,
                     verbose=0)
        # hats
        mu0_hat = tf.squeeze(temp_mu0(x_pred))
        mu1_hat = tf.squeeze(temp_mu1(x_pred))
        return mu0_hat, mu1_hat

    def fit(self,
            x, y, w):

        if self.method == 'rf':
            # hats
            mu0_hat, mu1_hat = self.compute_hats_rf(x, y, w, x)
            # pseudo-outcomes
            pseudo_outcomes = w * (y - mu0_hat) + (1 - w) * (mu1_hat - y)
            # 3: fit tau
            self.tau_model.fit(x, pseudo_outcomes)

        elif self.method == 'lasso':
            # hats
            mu0_hat, mu1_hat = self.compute_hats_lasso(x, y, w, x)
            # pseudo-outcomes
            pseudo_outcomes = w * (y - mu0_hat) + (1 - w) * (mu1_hat - y)
            # to poly
            x_poly = self.poly.fit_transform(x)
            # 3: fit tau
            self.tau_model.fit(x_poly, pseudo_outcomes)

        elif self.method == 'nn':
            # hats
            mu0_hat, mu1_hat = self.compute_hats_nn(x, y, w, x)
            # pseudo-outcomes
            pseudo_outcomes = w * (y - mu0_hat) + (1 - w) * (mu1_hat - y)
            # 3: fit tau
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
            raise ValueError('Base learner method not (or not correctly) specified. Can be "rf", "lasso" or "nn".')

        return predictions


class PWLearner:
    def __init__(self, method):
        self.method = method
        self.name = f"PWLearner, {self.method}"

        if self.method == 'rf':
            self.tau_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)

        elif self.method == 'lasso':
            self.tau_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.poly = PolynomialFeatures(degree=DEGREE_POLYNOMIALS, interaction_only=False, include_bias=False)

        elif self.method == 'nn':
            self.tau_model = clone_nn_regression(NN_SEQUENTIAL)

        else:
            raise ValueError('Base learner method not (or not correctly) specified. Can be "rf", "lasso" or "nn".')

    @staticmethod
    def compute_hats_rf(x_fit, w_fit, x_pred):
        # set models
        temp_ex = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=RANDOM)
        # fit
        temp_ex.fit(x_fit, w_fit)
        # hats
        probs = temp_ex.predict_proba(x_pred)[:, 1]
        # clipping probabilities
        np.clip(probs, MIN_CLIP, MAX_CLIP, out=probs)
        return probs

    def compute_hats_lasso(self, x_fit, w_fit, x_pred):
        # poly transformation
        x_poly_fit = self.poly.fit_transform(x_fit)
        x_poly_pred = self.poly.fit_transform(x_pred)
        # set models
        temp_ex = LogisticRegressionCV(cv=KFold(K_FOLDS), penalty='l1', solver='saga', tol=TOLERANCE,
                                       random_state=RANDOM,
                                       max_iter=MAX_ITER)
        # fit
        temp_ex.fit(x_poly_fit, w_fit)
        # hats
        probs = temp_ex.predict_proba(x_poly_pred)[:, 1]
        # clipping probabilities
        np.clip(probs, MIN_CLIP, MAX_CLIP, out=probs)
        return probs

    @staticmethod
    def compute_hats_nn(x_fit, w_fit, x_pred):
        # to tensor
        x_fit = tf.convert_to_tensor(x_fit)
        w_fit = tf.convert_to_tensor(w_fit)
        x_pred = tf.convert_to_tensor(x_pred)
        # set models
        temp_ex = clone_nn_classification(NN_SEQUENTIAL)
        # fit
        temp_ex.fit(x_fit, w_fit,
                    batch_size=BATCH_SIZE,
                    epochs=N_EPOCHS,
                    callbacks=CALLBACK,
                    validation_split=VALIDATION_SPLIT,
                    verbose=0)
        # hats
        probs = tf.squeeze(keras.activations.sigmoid(temp_ex(x_pred)))
        # clipping probabilities
        probs = tf.clip_by_value(probs, MIN_CLIP, MAX_CLIP)
        return probs

    def fit(self,
            x, y, w):
        if self.method == 'rf':
            # hats
            probs = self.compute_hats_rf(x, w, x)
            # pseudo-outcomes
            pseudo_outcomes = (w / probs - (1 - w) / (1 - probs)) * y
            # 3: fit tau
            self.tau_model.fit(x, pseudo_outcomes)

        elif self.method == 'lasso':
            # hats
            probs = self.compute_hats_lasso(x, w, x)
            # pseudo-outcomes
            pseudo_outcomes = tf.squeeze((w / probs - (1 - w) / (1 - probs)) * y)
            # to poly
            x_poly = self.poly.fit_transform(x)
            # 3: fit tau
            self.tau_model.fit(x_poly, pseudo_outcomes)

        elif self.method == 'nn':
            # hats
            probs = self.compute_hats_nn(x, w, x)
            # pseudo-outcomes
            pseudo_outcomes = (w / probs - (1 - w) / (1 - probs)) * y
            # 3: fit tau
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
            raise ValueError('Base learner method not (or not correctly) specified. Can be "rf", "lasso" or "nn".')

        return predictions


class ULearner:
    def __init__(self, method):
        self.method = method
        self.name = f"ULearner, {self.method}"

        if self.method == 'rf':
            self.tau_model = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH,
                                                   random_state=RANDOM)

        elif self.method == 'lasso':
            self.tau_model = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
            self.poly = PolynomialFeatures(degree=DEGREE_POLYNOMIALS, interaction_only=False, include_bias=False)

        elif self.method == 'nn':
            self.tau_model = clone_nn_regression(NN_SEQUENTIAL)

        else:
            raise ValueError('Base learner method not (or not correctly) specified. Can be "rf", "lasso" or "nn".')

    @staticmethod
    def compute_hats_rf(x_fit, y_fit, w_fit, x_pred):
        # set models
        temp_mux = RandomForestRegressor(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=RANDOM)
        temp_ex = RandomForestClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=RANDOM)
        # fit
        temp_mux.fit(x_fit, y_fit)
        temp_ex.fit(x_fit, w_fit)
        # hats
        mux_hat = temp_mux.predict(x_pred)
        probs = temp_ex.predict_proba(x_pred)[:, 1]
        # clipping probabilities
        np.clip(probs, MIN_CLIP, MAX_CLIP, out=probs)
        return mux_hat, probs

    def compute_hats_lasso(self, x_fit, y_fit, w_fit, x_pred):
        # poly transformation
        x_poly_fit = self.poly.fit_transform(x_fit)
        x_poly_pred = self.poly.fit_transform(x_pred)
        # set models
        temp_mux = LassoCV(cv=K_FOLDS, tol=TOLERANCE, random_state=RANDOM, max_iter=MAX_ITER)
        temp_ex = LogisticRegressionCV(cv=KFold(K_FOLDS), penalty='l1', solver='saga', tol=TOLERANCE,
                                       random_state=RANDOM,
                                       max_iter=MAX_ITER)
        # fit
        temp_mux.fit(x_poly_fit, y_fit)
        temp_ex.fit(x_poly_fit, w_fit)
        # hats
        mux_hat = temp_mux.predict(x_poly_pred)
        probs = temp_ex.predict_proba(x_poly_pred)[:, 1]
        # clipping probabilities
        np.clip(probs, MIN_CLIP, MAX_CLIP, out=probs)
        return mux_hat, probs

    @staticmethod
    def compute_hats_nn(x_fit, y_fit, w_fit, x_pred):
        # to tensor
        x_fit = tf.convert_to_tensor(x_fit)
        y_fit = tf.convert_to_tensor(y_fit)
        w_fit = tf.convert_to_tensor(w_fit)
        x_pred = tf.convert_to_tensor(x_pred)
        # set models
        temp_mux = clone_nn_regression(NN_SEQUENTIAL)
        temp_ex = clone_nn_classification(NN_SEQUENTIAL)
        # fit
        temp_mux.fit(x_fit, y_fit,
                     batch_size=BATCH_SIZE,
                     epochs=N_EPOCHS,
                     callbacks=CALLBACK,
                     validation_split=VALIDATION_SPLIT,
                     verbose=0)
        temp_ex.fit(x_fit, w_fit,
                    batch_size=BATCH_SIZE,
                    epochs=N_EPOCHS,
                    callbacks=CALLBACK,
                    validation_split=VALIDATION_SPLIT,
                    verbose=0)
        # hats
        mux_hat = tf.squeeze(temp_mux(x_pred))
        probs = tf.squeeze(keras.activations.sigmoid(temp_ex(x_pred)))
        # clipping probabilities
        probs = tf.clip_by_value(probs, MIN_CLIP, MAX_CLIP)
        return mux_hat, probs

    def fit(self,
            x, y, w):

        if self.method == 'rf':
            # hats
            mux_hat, probs = self.compute_hats_rf(x, y, w, x)
            # pseudo-outcomes
            pseudo_outcomes = (y - mux_hat) / (w - probs)
            # 3: fit tau
            self.tau_model.fit(x, pseudo_outcomes)

        elif self.method == 'lasso':
            # hats
            mux_hat, probs = self.compute_hats_lasso(x, y, w, x)
            # pseudo-outcomes
            pseudo_outcomes = (y - mux_hat) / (w - probs)
            # to poly
            x_poly = self.poly.fit_transform(x)
            # 3: fit tau
            self.tau_model.fit(x_poly, pseudo_outcomes)

        elif self.method == 'nn':
            # hats
            mux_hat, probs = self.compute_hats_nn(x, y, w, x)
            # pseudo-outcomes
            pseudo_outcomes = (y - mux_hat) / (w - probs)
            # 3: fit tau
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
            raise ValueError('Base learner method not (or not correctly) specified. Can be "rf", "lasso" or "nn".')

        return predictions
