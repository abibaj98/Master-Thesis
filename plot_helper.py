import jsonpickle
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import rankdata

# helper function to read json file
def open_json(name):
    f = open(name, 'r')
    json_str = f.read()
    file = jsonpickle.decode(json_str)
    return file

# helper function to make data plot ready
def make_plot_ready(data_set, setting):
    # make them dataframes
    df1 = pd.DataFrame(data_set[0])
    df2 = pd.DataFrame(data_set[1])
    df3 = pd.DataFrame(data_set[2])
    # add metalearner names
    column_names = ['T-Learner', 'S-Learner', 'X-Learner', 'R-Learner', 'DR-Learner', 'RA-Learner', 'PW-Learner',
                    'U-Learner',
                    'SampleSize']
    df1.columns = column_names
    df2.columns = column_names
    df3.columns = column_names
    # wide format -> long format
    df1 = df1.melt('SampleSize', var_name='MetaLearner', value_name='MSE')
    df2 = df2.melt('SampleSize', var_name='MetaLearner', value_name='MSE')
    df3 = df3.melt('SampleSize', var_name='MetaLearner', value_name='MSE')
    # add new column indicating baselearner
    df1['BaseLearner'] = 'Random Forest'
    df2['BaseLearner'] = 'Lasso-Based'
    df3['BaseLearner'] = 'Neural Network'
    # concatenate
    df_1 = pd.concat([df1, df2, df3], ignore_index=True)
    # column setting
    df_1['Setting'] = setting
    # using dictionary to convert specific columns
    dictionary = {'SampleSize': int,
                  'MetaLearner': 'category',
                  'Setting': 'category'
                  }
    df_1 = df_1.astype(dictionary)
    return df_1

# helper function to make data plot ready (rankings)
def make_rank_ready(d1):
    rf = np.zeros((0, 9))
    lasso = np.zeros((0, 9))
    nn = np.zeros((0, 9))
    # for random forest (=0)
    for i in range(1, 25):
        rf = np.vstack((rf, d1["data{0}".format(i)][0][:, 0:9]))
    # for lasso (=1)
    for i in range(1, 25):
        lasso = np.vstack((lasso, d1["data{0}".format(i)][1][:, 0:9]))
    # for lasso (=1)
    for i in range(1, 25):
        nn = np.vstack((nn, d1["data{0}".format(i)][2][:, 0:9]))
    # random forest
    rf_500 = rf[rf[:, 8] == 500][:, 0:8]
    rf_1000 = rf[rf[:, 8] == 1000][:, 0:8]
    rf_2000 = rf[rf[:, 8] == 2000][:, 0:8]
    rf_5000 = rf[rf[:, 8] == 5000][:, 0:8]
    # random forest
    lasso_500 = lasso[lasso[:, 8] == 500][:, 0:8]
    lasso_1000 = lasso[lasso[:, 8] == 1000][:, 0:8]
    lasso_2000 = lasso[lasso[:, 8] == 2000][:, 0:8]
    lasso_5000 = lasso[lasso[:, 8] == 5000][:, 0:8]
    # random forest
    nn_500 = nn[nn[:, 8] == 500][:, 0:8]
    nn_1000 = nn[nn[:, 8] == 1000][:, 0:8]
    nn_2000 = nn[nn[:, 8] == 2000][:, 0:8]
    nn_5000 = nn[nn[:, 8] == 5000][:, 0:8]
    # ranks
    ranks_rf_500 = rankdata(rf_500, axis=1)
    ranks_rf_1000 = rankdata(rf_1000, axis=1)
    ranks_rf_2000 = rankdata(rf_2000, axis=1)
    ranks_rf_5000 = rankdata(rf_5000, axis=1)
    ranks_lasso_500 = rankdata(lasso_500, axis=1)
    ranks_lasso_1000 = rankdata(lasso_1000, axis=1)
    ranks_lasso_2000 = rankdata(lasso_2000, axis=1)
    ranks_lasso_5000 = rankdata(lasso_5000, axis=1)
    ranks_nn_500 = rankdata(nn_500, axis=1)
    ranks_nn_1000 = rankdata(nn_1000, axis=1)
    ranks_nn_2000 = rankdata(nn_2000, axis=1)
    ranks_nn_5000 = rankdata(nn_5000, axis=1)
    # mean ranks
    rf_rankmed_500 = np.mean(ranks_rf_500, axis=0)
    rf_rankmed_1000 = np.mean(ranks_rf_1000, axis=0)
    rf_rankmed_2000 = np.mean(ranks_rf_2000, axis=0)
    rf_rankmed_5000 = np.mean(ranks_rf_5000, axis=0)
    lasso_rankmed_500 = np.mean(ranks_lasso_500, axis=0)
    lasso_rankmed_1000 = np.mean(ranks_lasso_1000, axis=0)
    lasso_rankmed_2000 = np.mean(ranks_lasso_2000, axis=0)
    lasso_rankmed_5000 = np.mean(ranks_lasso_5000, axis=0)
    nn_rankmed_500 = np.mean(ranks_nn_500, axis=0)
    nn_rankmed_1000 = np.mean(ranks_nn_1000, axis=0)
    nn_rankmed_2000 = np.mean(ranks_nn_2000, axis=0)
    nn_rankmed_5000 = np.mean(ranks_nn_5000, axis=0)
    # stack
    rf_rankmed = np.vstack((rf_rankmed_500.reshape(1, 8), rf_rankmed_1000.reshape(1, 8), rf_rankmed_2000.reshape(1, 8),
                            rf_rankmed_5000.reshape(1, 8)))
    lasso_rankmed = np.vstack((lasso_rankmed_500.reshape(1, 8), lasso_rankmed_1000.reshape(1, 8),
                               lasso_rankmed_2000.reshape(1, 8), lasso_rankmed_5000.reshape(1, 8)))
    nn_rankmed = np.vstack((nn_rankmed_500.reshape(1, 8), nn_rankmed_1000.reshape(1, 8), nn_rankmed_2000.reshape(1, 8),
                            nn_rankmed_5000.reshape(1, 8)))
    # to pandas
    rf_rankmed = pd.DataFrame(rf_rankmed)
    rf_rankmed['SampleSize'] = [500, 1000, 2000, 5000]
    lasso_rankmed = pd.DataFrame(lasso_rankmed)
    lasso_rankmed['SampleSize'] = [500, 1000, 2000, 5000]
    nn_rankmed = pd.DataFrame(nn_rankmed)
    nn_rankmed['SampleSize'] = [500, 1000, 2000, 5000]
    # column names
    column_names = ['T-Learner', 'S-Learner', 'X-Learner', 'R-Learner', 'DR-Learner', 'RA-Learner',
                    'PW-Learner',
                    'U-Learner',
                    'SampleSize']
    rf_rankmed.columns = column_names
    lasso_rankmed.columns = column_names
    nn_rankmed.columns = column_names
    # finally melt
    melted_rf = rf_rankmed.melt('SampleSize', var_name='MetaLearner', value_name='MeanRank')
    melted_rf['SampleSize'] = melted_rf['SampleSize'].astype('category')
    melted_lasso = lasso_rankmed.melt('SampleSize', var_name='MetaLearner', value_name='MeanRank')
    melted_lasso['SampleSize'] = melted_lasso['SampleSize'].astype('category')
    melted_nn = nn_rankmed.melt('SampleSize', var_name='MetaLearner', value_name='MeanRank')
    melted_nn['SampleSize'] = melted_nn['SampleSize'].astype('category')

    return melted_rf, melted_lasso, melted_nn

def ihdp_make_plot_ready():
    ihdp = open_json("final_results_final/results_ihdp_100run(s)_final.json")
    ihdp_rf = ihdp[0]  # RF
    ihdp_lm = ihdp[1]  # lasso-based
    ihdp_nn = ihdp[2]  # NN
    # make df
    ihdp_rf = pd.DataFrame(ihdp_rf)
    ihdp_lm = pd.DataFrame(ihdp_lm)
    ihdp_nn = pd.DataFrame(ihdp_nn)
    # column names
    columns = ['T', 'S', 'X', 'R', 'DR', 'RA', 'PW',
               'U']
    ihdp_rf.columns = columns
    ihdp_lm.columns = columns
    ihdp_nn.columns = columns
    # melt
    ihdp_rf = ihdp_rf.melt(var_name='MetaLearner', value_name='MSE')
    ihdp_lm = ihdp_lm.melt(var_name='MetaLearner', value_name='MSE')
    ihdp_nn = ihdp_nn.melt(var_name='MetaLearner', value_name='MSE')
    # names base-learner
    ihdp_rf['BaseLearner'] = 'RandomForest'
    ihdp_lm['BaseLearner'] = 'Lasso-Based'
    ihdp_nn['BaseLearner'] = 'NeuralNetwork'
    # concatenate
    ihdp_all = pd.concat([ihdp_rf, ihdp_lm, ihdp_nn])

    return ihdp_all

def ihdp_make_rank_ready(learner_names):
    ihdp = open_json("final_results_final/results_ihdp_100run(s)_final.json")
    # rank
    ihdp_rank_rf = rankdata(ihdp[0], axis=1)
    ihdp_rank_lasso = rankdata(ihdp[1], axis=1)
    ihdp_rank_nn = rankdata(ihdp[2], axis=1)
    # meanranks
    ihdp_meanranks_rf = np.mean(ihdp_rank_rf, axis=0).reshape(1, 8)
    ihdp_meanranks_lasso = np.mean(ihdp_rank_lasso, axis=0).reshape(1, 8)
    ihdp_meanranks_nn = np.mean(ihdp_rank_nn, axis=0).reshape(1, 8)
    # to df
    meanranks_rf = pd.DataFrame(ihdp_meanranks_rf)
    meanranks_lasso = pd.DataFrame(ihdp_meanranks_lasso)
    meanranks_nn = pd.DataFrame(ihdp_meanranks_nn)
    # column names
    meanranks_rf.columns = learner_names
    meanranks_lasso.columns = learner_names
    meanranks_nn.columns = learner_names
    # concat
    meanranks_ihdp = pd.concat([meanranks_rf, meanranks_lasso, meanranks_nn], axis=0)
    meanranks_ihdp['BaseLearner'] = ['RandomForest', 'Lasso-Based', 'NeuralNetwork']
    meanranks_ihdp['BaseLearner'] = meanranks_ihdp['BaseLearner'].astype('category')
    # melt
    meanranks_ihdp = meanranks_ihdp.melt('BaseLearner', var_name='MetaLearner', value_name='MeanRank')

    return meanranks_ihdp
