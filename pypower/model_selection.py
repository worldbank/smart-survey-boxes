'''
This module is for evaluating model perfomance
'''

import itertools
import multiprocessing
import random

import sys
import traceback
from datetime import datetime

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score

from pypower import data_utils as ut
from pypower import prediction_models as pred
from pypower import preprocessing as prep


def batch_evaluation_of_the_box_models(config_obj=None, pooled=False, inserted=True, sample_thres=500000):
    """
    Validates out of the box models such as random forest.
    :param df:
    :return:
    """
    # read in the data
    file_sms_v2 = config_obj.get_processed_data_dir() + 'sms_v2.csv'
    results_dir = config_obj.get_outputs_dir()

    cols_to_use = ['box_id', 'psu', 'lon', 'lat', 'str_datetime_sent_hr', 'day_sent','hour_sent','month_sent',
                   'wk_day_sent', 'wk_end', 'event_type_num', 'event_type_str', 'data_source']

    df = pd.read_csv(file_sms_v2, usecols=cols_to_use)

    # drop missing and test events
    num_missing = len(df[df.event_type_str == 'missing'])
    print('Number of missing events...{} out of total {} in rectangular dataset'.format(num_missing, df.shape[0]))
    print('Discarding missing events...we dont need them for validation...')

    df = df[df.event_type_str != 'missing']

    if df.shape[0] > sample_thres:
        df = df.sample(frac=0.30)

    # whether to use inserted events or only observed_events
    if not inserted:
        print('Use only observed events....dropping inserted events')
        df = df[df.data_source == 'observed_event']

    # Features to use for prediction
    prediction_features = ['box_id', 'psu', 'lon', 'lat', 'hour_sent', 'month_sent', 'day_sent', 'wk_day_sent',
                           'wk_end']

    # Define models
    random_state = 1
    clfs = {# 'RF': RandomForestClassifier(n_estimators=350, n_jobs=-1,
            #                           random_state=random_state),
            'ETC': ExtraTreesClassifier(n_estimators=350, n_jobs=-1, criterion='gini')
            }

    # finally eveluate
    results = evaluate_out_of_the_box_models(model_list=clfs, data=df, features=prediction_features,
                                             target='event_type_num', output_dir=results_dir)

    output = []
    for model, cv_score in results.items():
        print("(%s) CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g"
              % (
              model, np.mean(cv_score) * 100, np.std(cv_score) * 100, np.min(cv_score) * 100, np.max(cv_score) * 100))

        output.append({'model': model, 'mean': np.mean(cv_score) * 100,
                      'std_dev': np.std(cv_score) * 100, 'min': np.min(cv_score) * 100,
                      'max': np.max(cv_score) * 100})

    # save results to file
    df = pd.DataFrame(output)
    df.to_csv(results_dir + 'out_of_the_box_model_eval_res.csv', index=False)


def evaluate_out_of_the_box_models(model_list=None, data='df', features=None,
                                   target='event_type_num', output_dir=None):
    """
    Evaluates out of the box models and picks the best model based on accuracy
    using k-fold.
    :param model_list: dict-A list of models to try
    :param df: The data to test
    :param features:
    :param target: The variable to predict
    :return: A dict object with model name and accuracy
    """
    X = data[features].values
    y = data[target].values

    results = {}

    for nm, clf in model_list.items():
        print ('Working on ...%s' %nm)

        scores = cross_val_score(clf, X, y, cv=3)
        results[nm] = scores

        # Feature importance
        if nm != 'LR':
            clf.fit(X, y)
            feat_imp = pd.Series(clf.feature_importances_, features).sort_values(ascending=False)
            # print(feat_imp)
            feat_imp.plot(kind='bar')
            plt.ylabel('Feature Importance Score')
            figure = plt.gcf()  # get current figure
            figure.set_size_inches(10, 10)
            plot_file_name = output_dir + nm + '_feat_imp.png'
            figure.savefig(plot_file_name)

    return results


def batch_evaluation_imputation_nearest_neighbor(file_sms_v2=None, box_metadata_file=None,
                                                 model_params=None, eval_params=None):
    """
    This just evaluates the nearest box predictor with a temporal window
    :param df:
    :param use_random:
    :return: a list with accuracy for each PSU
    """
    # read in the data
    cols_to_use = ['box_id', 'psu', 'lon', 'lat', 'str_datetime_sent_hr', 'hour_sent', 'event_type_str']

    df = pd.read_csv(file_sms_v2, usecols=cols_to_use, parse_dates=['str_datetime_sent_hr'])

    df.rename(columns={'str_datetime_sent_hr': 'datetime_sent_hr'}, inplace=True)

    df = df[df.event_type_str != 'missing'] # Keep only non-missing values for evaluation

    # list of all boxes
    box_list = list(df.box_id.unique())
    random_boxes = random.sample(box_list, eval_params['num_boxes']) # select only a few boxes

    # dict object to hold all results
    acc_all = {}

    i = 0
    for bx in random_boxes:
        try:
            # preselect data based on box location
            bx_df = df[df.box_id == bx]

            box_xy = [bx_df['lat'].iloc[0], bx_df['lon'].iloc[0]]
            neighbors = ut.k_nearest_boxes(box_metadata_file=box_metadata_file, box_id=bx, box_lat_lon=box_xy, k=10)
            neighbors.append(bx)

            df_bx = df[df['box_id'].isin(neighbors)]

            if i % 10 == 0:
                print('{} boxes processed....'.format(i))

            results = evaluate_imputation_nearest_neighbor_model(all_data=df_bx, test_box_id=bx,
                                                                 prop_test=eval_params['num_tests'],
                                                                 min_test_cases=eval_params['min_cases'],
                                                                 max_test_cases=eval_params['min_cases'],
                                                                 predictor_params=model_params,
                                                                 box_file=box_metadata_file)

            acc_all[bx] = results
            i += 1
        except Exception as e:
            desired_trace = traceback.format_exc(sys.exc_info())
            print(desired_trace)
            continue

    return acc_all


def evaluate_imputation_nearest_neighbor_model(all_data=None, test_box_id=1000,
                                               prop_test=0.1, predictor_params=None,
                                               box_file=None, min_test_cases=5, max_test_cases=100):
    """
    Tests accuracy of nearest neighbor predictor for a specific box
    :param all_data:
    :param test_id:
    :param prop_test:
    :param window:
    :return:
    """
    # ------------PRELIMINARY SET UP-----------------------------------
    data_test_bx = all_data[all_data.box_id == test_box_id]  # select data for this box only

    event_dates = list(data_test_bx.datetime_sent_hr)

    # Randomly select test events and remove them from training data
    num_tests = int(data_test_bx.shape[0] * prop_test)

    # skip box if test cases are less than 10
    if num_tests < min_test_cases:
        return np.nan

    if num_tests > max_test_cases:
        num_tests = max_test_cases

    test_dates = random.sample(event_dates, num_tests)

    to_keep = list(set(event_dates) - set(test_dates))

    train_df = all_data[all_data['datetime_sent_hr'].isin(to_keep)]

    test_df = all_data[all_data['datetime_sent_hr'].isin(test_dates)]


    # ----------------CREATE MODEL OBJECT-----------------------------
    clf = pred.ImputationNearestNeighbor(data=train_df,
                                         neighbors=predictor_params['neighbors'],
                                         how=predictor_params['how'],
                                         time_window=predictor_params['time-window'],
                                         direction=predictor_params['direction']
                                         )

    clf.generate_box_metadata(box_file=box_file)
    box_lat_lon = [data_test_bx[data_test_bx.box_id==test_box_id].lat.values[0],
                   data_test_bx[data_test_bx.box_id == test_box_id].lon.values[0]]

    # --------------TEST MODEL PERFOMANCE---------------------------
    results = imputation_nearest_neigbor_model_accuracy(model_object=clf, test_data=test_df, box_id=test_box_id,
                                                        xy=box_lat_lon)

    return results


def imputation_nearest_neigbor_model_accuracy(model_object=None, test_data=None, box_id=90, xy=None):
    """
    Returns accuracy after making predictions for the test data.
    :param model_object: The prediction model
    :param test_data: test data
    :param box_id:
    :param xy:
    :return:
    """
    # DO THE TESTS
    correct = 0
    tot = 0

    for index, row in test_data.iterrows():
        # Retrieve actual value
        actual_event = row['event_type_str']

        # test date
        test_date = row['datetime_sent_hr']

        # make prediction
        predicted = model_object.predict(prediction_date=test_date, box_id=box_id, target_loc=xy)

        if predicted == actual_event:
            correct += 1

        tot += 1

    return correct / tot * 100


def param_selection_imputation_nearest_neigbor_model(results=None, output_file = None):
    """
    Summarises results and
    :param results:
    :return:
    """
    # -----------CREATE DATAFRAME FROM RESULTS-------------------------------------
    row_list = []

    for k, v in results.items():
        meta_data = {'window_len':k[0],'neighbors': k[1]}

        for b_id, acc in v.items():
            data_dict = dict()
            data_dict['box_id'] = b_id
            data_dict['accuracy'] = acc
            data_dict.update(meta_data)

            row_list.append(data_dict)

    # -----------SAVE RESULTS FOR ANALYSIS AND MODEL SELECTION----------------------------------------
    df = pd.DataFrame(row_list)
    df = df[['box_id', 'neighbors', 'window_len', 'accuracy']]
    df.sort_values(by='box_id').to_csv(output_file,index=False)
    print('Successfully saved param settings.....')
    return df


def evaluate_nearest_neighbor(configs=None, debug_mode=False):
    """
    Simply budnles togather the code for completely evaluating NN
    :return:
    """

    # -----SET UP DATA SOURCES----------------------------------
    sms_v2 = configs.get_processed_data_dir() + 'sms_v2.csv'
    boxes_file = configs.get_data_dir() + 'Boxes.csv'
    output_dir = configs.get_outputs_dir()

    # ----------- EVALUATE IMPUTATION-NEAREST NEIGHBOR----------
    neighbors = [0, 1, 3]
    window = [7, 14]
    window_neighbor_comb = [x for x in itertools.product(window, neighbors)]

    # parameter settings
    params_model = {'neighbors': 1, 'time-window': 7, 'direction': 'both', 'how': 'frequent'}

    num_boxes = 100
    if debug_mode:
        num_boxes = 50

    params_eval = {'num_tests': 0.01, 'min_cases': 10, 'max_cases': 50, 'num_boxes': num_boxes}  # hold out 10 percent of the data

    all_res = {}

    # test different combination of parameter
    start = datetime.now()
    print('DOING PARAMETER OPTIMISATION FOR NEAREST NEIGHBOR MODEL....')
    for (w, n) in window_neighbor_comb:
        print('Neighbors--> {}, window length-->{} days'.format(n, w))
        params_model['time-window'] = w
        params_model['neighbors'] = n
        res = batch_evaluation_imputation_nearest_neighbor(file_sms_v2=sms_v2, box_metadata_file=boxes_file,
                                                           model_params=params_model, eval_params=params_eval)
        all_res[(w, n)] = res

        # save results for each w,m
        results_file = output_dir + 'nn_eval_res.csv'
        param_selection_imputation_nearest_neigbor_model(results=all_res, output_file=results_file)

    end = datetime.now()
    time_taken = (end - start).total_seconds() / 60
    print('PARAMETER OPTIMISATION SUCCESSFULLY DONE, TOOK {0:.2f} minutes'.format(time_taken))
    sys.exit()


if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver', force=True)
    # ----------SET UP-----------------------------------------
    config = prep.Configurations(platform='mac')
    config.debug_mode = False

    # -----------EVALUATE OUT OF THE BOX MODELS----------------
    batch_evaluation_of_the_box_models(config_obj=config, pooled=False, inserted=True)

    # -----------EVALUATE NEAREST NEIGHBOR----------------------
    #evaluate_nearest_neighbor(configs=config, debug_mode=config.debug_mode)


