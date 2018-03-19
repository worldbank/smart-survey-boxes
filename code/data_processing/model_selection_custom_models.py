'''
This module is for evaluating model perfomance
'''
import linecache
import itertools
import multiprocessing
import random
import sys
import os
import traceback
from datetime import datetime
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

import data_utils as ut
import prediction_models as pred
import data_processing as prep

# fix prediction features for now
#PREDICTION_FEATURES = ['box_id', 'psu', 'lon', 'lat', 'hour_sent', 'month_sent', 'day_sent', 'wk_day_sent', 'wk_end']

# Try other features
PREDICTION_FEATURES = ['lon', 'lat', 'hour_sent', 'month_sent', 'day_sent', 'wk_day_sent', 'wk_end']


def extract_test_box_metadata(config_obj=None, test_box_id=None, model_params=None):
    """
    For the sake of detailed reporting of evaluation results, this extracts some basic metadata about the box under
    evaluation.
    :param config_obj: config object which has set up details
    :param test_box_id:
    :param model_params:
    :param df_bx: Box data which is going into testing
    :return:
    """
    # -----------GET EVALUATION DATA----------------------------------------------#
    df_non_missing, df_missing, df_invalid = get_data_to_evaluate(config_obj=config_obj)

    # dict to hold the details
    test_box_details = {}

    # log number of missing events for each box
    num_missing = df_missing[df_missing.box_id == test_box_id].shape[0]
    num_invalid = df_invalid[df_invalid.box_id == test_box_id].shape[0]
    non_missing = df_non_missing[df_non_missing.box_id == test_box_id].shape[0]
    bx_df = df_non_missing[df_non_missing.box_id == test_box_id]

    # add box details to thes objects
    test_box_details['total_non_missing_events'] = df_non_missing.shape[0]
    test_box_details['num_invalid_events'] = num_invalid
    test_box_details['num_missing_events'] = num_missing
    test_box_details['num_non_missing_events'] = non_missing
    test_box_details['psu'] = bx_df.iloc[0].psu
    test_box_details['region'] = bx_df.iloc[0].region
    test_box_details['window_length'] = model_params['time-window']
    test_box_details['neighbors'] = model_params['neighbors']


    return test_box_details


def print_exception():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


def update_results_object(obj=None, missing=None, invalid=None, psu_id=None, neighbors=None, window_len=None,
                          non_missing=None):
    return obj._replace(num_missing_events=missing, num_invalid_events=invalid, psu=psu_id, neighbors=neighbors,
                        window_length=window_len, num_non_missing_events=non_missing)


def get_data_to_evaluate(config_obj=None):
    """
    Simply reads in the sms_rect_hr.csv file, removes missing events since we don't need them for evaluation.
    Also, returns df of missing events for use in reporting.
    :param config_obj:
    :return:
    """
    # the data file
    file_sms_v2 = os.path.join(config_obj.get_processed_data_dir(), "sms_rect_hr.csv")

    # read in the data
    cols_to_use = ['box_id', 'psu', 'region', 'lon', 'lat', 'str_datetime_sent_hr', 'day_sent', 'hour_sent',
                   'month_sent',
                   'wk_day_sent', 'wk_end', 'event_type_num', 'event_type_str', 'power_state', 'data_source']

    df = pd.read_csv(file_sms_v2, usecols=cols_to_use, parse_dates=['str_datetime_sent_hr'])

    # rename columns for convenience
    df.rename(columns={'str_datetime_sent_hr': 'datetime_sent_hr'}, inplace=True)

    # remove missing events
    df_non_missing = df[df['power_state'].isin([1, 0])]  # Keep only non-missing values for evaluation

    # We need missing events for reporting
    df_missing = df[df.power_state == -1]
    df_invalid = df[df.power_state == 99]

    return df_non_missing, df_missing, df_invalid


def prepare_data_for_training_testing(data=None, box_id=1000, prop_test=0.1, min_test_cases=5, max_test_cases=100):
    """
    Preliminary set up for testing and evaluation.
    :return:
    """
    data_test_bx = data[data.box_id == box_id]  # select data for this box only

    # all dates in the test box
    test_box_event_dates = list(data_test_bx.datetime_sent_hr)

    # Randomly select test events and remove them from training data
    num_tests = int(data_test_bx.shape[0] * prop_test)

    # skip box if test cases are less than 10
    if num_tests < min_test_cases:
        return None

    if num_tests > max_test_cases:
        num_tests = max_test_cases

    test_event_dates = random.sample(test_box_event_dates, num_tests)

    # Remove all test box data from all_data
    all_data_without_test_box = data[data.box_id != box_id]

    # Now, we only need to remove the exact test events
    to_keep = list(set(test_box_event_dates) - set(test_event_dates))

    # Within the test box, we can keep the non-test events
    test_box_to_keep = data_test_bx[data_test_bx['datetime_sent_hr'].isin(to_keep)]

    # Finally, our training dataset is sms2 minus test events in test box
    train_df = all_data_without_test_box.append(test_box_to_keep)

    test_df = data_test_bx[data_test_bx['datetime_sent_hr'].isin(test_event_dates)]

    # check training and test data
    check_training_and_test_data(train_df=train_df, test_data=test_df, test_box_id=box_id,
                                 test_dates=test_event_dates)

    return train_df, test_df


def preselect_data_based_on_test_box(df_all=None, box_id=None, k=10, box_metadata_file=None):
    """
    To speed up evaluation, instead of passing the whole dataset when evaluating one box,
    pass data for the nearest neighbors only.
    :return:
    """
    # preselect data based on box location
    bx_df = df_all[df_all.box_id == box_id]
    box_xy = [bx_df['lat'].iloc[0], bx_df['lon'].iloc[0]]
    neighbors = ut.k_nearest_boxes(box_metadata_file=box_metadata_file, box_id=box_id, box_lat_lon=box_xy, k=10)
    neighbors.append(box_id)

    df_bx = df_all[df_all['box_id'].isin(neighbors)]

    return df_bx


def save_metric_objects(metrics=None, out_file=None):
    """
    saves the results to file
    :param metrics:
    :param out_file:
    :return:
    """
    cols = ['psu', 'region', 'box_id', 'support', 'window_length', 'neighbors', 'model_name', 'model_type',
            'max_training_cases', 'mean_training_cases', 'median_training_cases', 'min_training_cases',
            'total_non_missing_events', 'num_invalid_events', 'num_missing_events', 'num_non_missing_events',
            'actual_0', 'actual_1', 'correct_pred_0', 'correct_pred_1', 'tot_pred_0', 'tot_pred_1', 'accuracy',
            'precision_0', 'precision_0_sc','precision_1', 'precision_1_sc','avg_precision', 'avg_precision_sc_macro',
            'avg_precision_sc_micro', 'recall_0', 'recall_0_sc','recall_1', 'recall_1_sc', 'avg_recall_sc_macro',
            'avg_recall_sc_micro','f1_score_0', 'f1_score_0_sc', 'f1_score_1', 'f1_score_1_sc', 'avg_f1_score',
            'avg_f1_score_sc_macro', 'avg_f1_score_sc_micro']

    df = pd.DataFrame(metrics)
    #df = df[cols]
    df.to_csv(out_file, index=False)


def check_training_and_test_data(train_df=None, test_box_id=None, test_data=None, test_dates=None):
    """
    Checks training and test data to ensure they are no leaks.
    """
    train_df_test_box = train_df[train_df.box_id == test_box_id]

    leaked = 0
    for date in test_dates:
        if date in list(train_df_test_box.datetime_sent_hr):
            print('WAIT A MINUTE, HOW COME TEST EVENTS ARE STILL IN TRAINIGNG DATA')
            leaked += 1

    if leaked != 0:
        print('Wait, found some test events in training data')


def compute_metrics_power_state2(model_object: pred.ImputationNearestNeighbor, test_data=None, box_id=90, xy=None,
                                target='power_state', model_type='nn', box_metadata=None):
    """
    Returns accuracy after making predictions for the test data.
    :param model_object: The prediction model
    :param test_data: test data
    :param box_id:
    :param xy:
    :return:
    """
    # variables for computing metrics
    correct = 0
    tot = 0

    actual_1 = 0
    actual_0 = 0

    tot_pred_1 = 0
    correct_1 = 0

    tot_pred_0 = 0
    correct_0 = 0

    training_examples = []
    y_pred=[]
    y_true=[]
    for index, row in test_data.iterrows():
        # Retrieve actual value
        actual_event = int(row[target])
        # proportions
        if actual_event == 1:
            actual_1 += 1

        if actual_event == 0:
            actual_0 += 1

        # test date
        test_date = row['datetime_sent_hr']

        # make prediction
        if model_type == 'nn':
            predicted = model_object.predict(prediction_date=test_date, box_id=box_id, target_loc=xy, model_type='nn')
        elif model_type == 'out':
            test_X = row[model_object.prediction_features].values
            predicted = model_object.predict(prediction_date=test_date, box_id=box_id, target_loc=xy, model_type='out',
                                             test_X=test_X.reshape((1, test_X.shape[0])))
        elif model_type == 'major':
            predicted = model_object.predict(prediction_date=test_date, box_id=box_id, target_loc=xy,
                                             model_type='major')
        elif model_type == 'rand':
            predicted = model_object.predict(prediction_date=test_date, box_id=box_id, target_loc=xy,
                                             model_type='rand')
        if np.isnan(predicted):
            continue

        y_pred.append(predicted)
        y_true.append(actual_event)
        # update training examples for stats
        training_examples.append(model_object.num_training_examples)

        if predicted == 1:
            tot_pred_1 += 1
            if actual_event == 1:
                correct_1 += 1

        if predicted == 0:
            tot_pred_0 += 1
            if actual_event == 0:
                correct_0 += 1

        if predicted == actual_event:
            correct += 1

        tot += 1

    # metrics
    accuracy = correct / tot * 100
    precision_0, precision_1, recall_0, recall_1 = [0.0 for _ in range(4)]

    if tot_pred_1 > 0:
        precision_1 = correct_1 / tot_pred_1 * 100

    if actual_1 > 0:
        recall_1 = correct_1 / actual_1 * 100

    if tot_pred_0 > 0:
        precision_0 = correct_0 / tot_pred_0 * 100

    if actual_0 > 0:
        recall_0 = correct_0 / actual_0 * 100

    if precision_0 == 0 and recall_0 == 0:
       F1_0 = 0
    else:
        F1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)

    if precision_1 == 0 and recall_1 == 0:
        F1_1 = 0
    else:
        F1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)


    avg_f1 = np.mean([F1_0, F1_1])
    avg_prec = np.mean([precision_0, precision_1])

    metrics = pred.Results(box_id=box_id, model_type=model_type, model_name='etc', actual_1=actual_1, actual_0=actual_0,
                           accuracy=accuracy, tot_pred_0=tot_pred_0, neighbors='',
                           tot_pred_1=tot_pred_1,precision_1=precision_1, precision_0=precision_0, recall_1=recall_1,
                           recall_0=recall_0,support=tot, correct_1=correct_1, correct_0=correct_0,
                           max_training_cases=np.max(training_examples), min_training_cases=np.min(training_examples),
                           median_training_cases=np.median(training_examples), avg_f1=avg_f1, f1_0=F1_0, f1_1=F1_1,
                           mean_training_cases=np.mean(training_examples), avg_prec=avg_prec)

    # Add scikit-learn metrics
    metrics.precision_0_sc = precision_score(y_pred=y_pred, y_true=y_true, pos_label=0)
    metrics.precision_1_sc = precision_score(y_pred=y_pred, y_true=y_true, pos_label=1)
    metrics.avg_precision_sc_macro = precision_score(y_pred=y_pred, y_true=y_true, average='macro')
    metrics.avg_precision_sc_micro = precision_score(y_pred=y_pred, y_true=y_true, average='micro')


    metrics.recall_0_sc = recall_score(y_pred=y_pred, y_true=y_true, pos_label=0)
    metrics.recall_1_sc = recall_score(y_pred=y_pred, y_true=y_true, pos_label=1)
    metrics.avg_recall_sc_macro = recall_score(y_pred=y_pred, y_true=y_true, pos_label=1, average='macro')
    metrics.avg_recall_sc_micro = recall_score(y_pred=y_pred, y_true=y_true, pos_label=1, average='micro')

    metrics.f1_score_0_sc = f1_score(y_pred=y_pred, y_true=y_true, pos_label=0)
    metrics.f1_score_1_sc= f1_score(y_pred=y_pred, y_true=y_true, pos_label=1)
    metrics.avg_f1_score_sc_macro = f1_score(y_pred=y_pred, y_true=y_true, average='macro')
    metrics.avg_f1_score_sc_micro = f1_score(y_pred=y_pred, y_true=y_true, average='micro')

    for key, value in box_metadata.items():
        setattr(metrics, key, value)

    return metrics


def compute_metrics_power_state(model_object: pred.ImputationNearestNeighbor, test_data=None, box_id=90, xy=None,
                                target='power_state', model_type='nn'):
    """
    Returns accuracy after making predictions for the test data.
    :param model_object: The prediction model
    :param test_data: test data
    :param box_id:
    :param xy:
    :return:
    """
    # variables for computing metrics
    correct = 0
    tot = 0

    actual_1 = 0
    actual_0 = 0

    tot_pred_1 = 0
    correct_1 = 0

    tot_pred_0 = 0
    correct_0 = 0

    training_examples = []
    y_pred=[]
    y_true=[]
    for index, row in test_data.iterrows():
        # Retrieve actual value
        actual_event = int(row[target])
        # proportions
        if actual_event == 1:
            actual_1 += 1

        if actual_event == 0:
            actual_0 += 1

        # test date
        test_date = row['datetime_sent_hr']

        # make prediction
        if model_type == 'nn':
            predicted = model_object.predict(prediction_date=test_date, box_id=box_id, target_loc=xy, model_type='nn')
        elif model_type == 'out':
            test_X = row[model_object.prediction_features].values
            predicted = model_object.predict(prediction_date=test_date, box_id=box_id, target_loc=xy, model_type='out',
                                             test_X=test_X.reshape((1, test_X.shape[0])))
        elif model_type == 'major':
            predicted = model_object.predict(prediction_date=test_date, box_id=box_id, target_loc=xy,
                                             model_type='major')
        elif model_type == 'rand':
            predicted = model_object.predict(prediction_date=test_date, box_id=box_id, target_loc=xy,
                                             model_type='rand')
        if np.isnan(predicted):
            continue

        y_pred.append(predicted)
        y_true.append(actual_event)
        # update training examples for stats
        training_examples.append(model_object.num_training_examples)

        if predicted == 1:
            tot_pred_1 += 1
            if actual_event == 1:
                correct_1 += 1

        if predicted == 0:
            tot_pred_0 += 1
            if actual_event == 0:
                correct_0 += 1

        if predicted == actual_event:
            correct += 1

        tot += 1

    # metrics
    accuracy = correct / tot * 100
    precision_0, precision_1, recall_0, recall_1 = [0.0 for _ in range(4)]

    if tot_pred_1 > 0:
        precision_1 = correct_1 / tot_pred_1 * 100

    if actual_1 > 0:
        recall_1 = correct_1 / actual_1 * 100

    if tot_pred_0 > 0:
        precision_0 = correct_0 / tot_pred_0 * 100

    if actual_0 > 0:
        recall_0 = correct_0 / actual_0 * 100

    if precision_0 == 0 and recall_0 == 0:
       F1_0 = 0
    else:
        F1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)

    if precision_1 == 0 and recall_1 == 0:
        F1_1 = 0
    else:
        F1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)


    avg_f1 = np.mean([F1_0, F1_1])
    avg_prec = np.mean([precision_0, precision_1])

    metrics = pred.Results(box_id=box_id, model_type=model_type, model_name='etc', actual_1=actual_1, actual_0=actual_0,
                           accuracy=accuracy, tot_pred_0=tot_pred_0, neighbors='', tot_pred_1=tot_pred_1,
                           precision_1=precision_1, precision_0=precision_0, recall_1=recall_1, recall_0=recall_0,
                           support=tot, correct_1=correct_1, correct_0=correct_0,
                           max_training_cases=np.max(training_examples), min_training_cases=np.min(training_examples),
                           median_training_cases=np.median(training_examples), avg_f1=avg_f1, f1_0=F1_0, f1_1=F1_1,
                           mean_training_cases=np.mean(training_examples), avg_prec=avg_prec)

    # Add scikit-learn metrics
    metrics.precision_0_sc = precision_score(y_pred=y_pred, y_true=y_true, pos_label=0)
    metrics.precision_1_sc = precision_score(y_pred=y_pred, y_true=y_true, pos_label=1)
    metrics.avg_precision_sc_macro = precision_score(y_pred=y_pred, y_true=y_true, average='macro')
    metrics.avg_precision_sc_micro = precision_score(y_pred=y_pred, y_true=y_true, average='micro')


    metrics.recall_0_sc = recall_score(y_pred=y_pred, y_true=y_true, pos_label=0)
    metrics.recall_1_sc = recall_score(y_pred=y_pred, y_true=y_true, pos_label=1)
    metrics.avg_recall_sc_macro = recall_score(y_pred=y_pred, y_true=y_true, pos_label=1, average='macro')
    metrics.avg_recall_sc_micro = recall_score(y_pred=y_pred, y_true=y_true, pos_label=1, average='micro')

    metrics.f1_score_0_sc = f1_score(y_pred=y_pred, y_true=y_true, pos_label=0)
    metrics.f1_score_1_sc= f1_score(y_pred=y_pred, y_true=y_true, pos_label=1)
    metrics.avg_f1_score_sc_macro = f1_score(y_pred=y_pred, y_true=y_true, average='macro')
    metrics.avg_f1_score_sc_micro = f1_score(y_pred=y_pred, y_true=y_true, average='micro')

    return metrics


def nn_vs_out_of_the_box_model_parallel(all_data=None, test_box_id=1000,
                                             prop_test=0.1, predictor_params=None, test_box_meta=None,
                                             box_file=None, min_test_cases=5, max_test_cases=100):
    """
    Tests accuracy of nearest neighbor predictor vs a selected out of the box model for a single box
    :param all_data:
    :param test_id:
    :param prop_test:
    :param window:
    :param test_box_metadata: miscellenous metadata about the box
    :return:
    """
    # ------------PRELIMINARY SET UP-----------------------------------
    data_test_bx = all_data[all_data.box_id == test_box_id]  # select data for this box only
    train_df, test_df = prepare_data_for_training_testing(data=all_data, box_id=test_box_id,
                                                          min_test_cases=min_test_cases,
                                                          prop_test=prop_test, max_test_cases=max_test_cases)

    # ----------------CREATE MODEL OBJECT-----------------------------
    num_boxes = len(list(all_data.box_id.unique()))

    etc = ETC(n_estimators=100)
    clf = pred.ImputationNearestNeighbor(data=train_df, target=predictor_params['target'],
                                         neighbors=predictor_params['neighbors'],
                                         how=predictor_params['how'],
                                         time_window=predictor_params['time-window'],
                                         direction=predictor_params['direction'], out_of_box_model=etc,
                                         pred_features=PREDICTION_FEATURES)

    clf.generate_box_metadata(box_file=box_file)
    box_lat_lon = [data_test_bx[data_test_bx.box_id == test_box_id].lat.values[0],
                   data_test_bx[data_test_bx.box_id == test_box_id].lon.values[0]]

    # --------------TEST MODEL PERFOMANCE OUT OF THE BOX---------------------------------------------------
    results_out = compute_metrics_power_state2(model_object=clf, test_data=test_df, box_id=test_box_id, xy=box_lat_lon,
                                              model_type='out',box_metadata=test_box_meta)

    # --------------TEST MODEL PERFOMANCE NEAREST NEIGHBOR-------------------------------------------------
    results_nearest = compute_metrics_power_state2(model_object=clf, test_data=test_df, box_id=test_box_id,
                                                  xy=box_lat_lon, model_type='nn',box_metadata=test_box_meta)

    # --------------TEST MODEL PERFOMANCE MAJORITY CLASSIFIER-------------------------------------------------
    results_majority = compute_metrics_power_state2(model_object=clf, test_data=test_df, box_id=test_box_id,
                                                  xy=box_lat_lon, model_type='major',box_metadata=test_box_meta)

    # --------------TEST MODEL PERFOMANCE RANDOM-------------------------------------------------
    results_random = compute_metrics_power_state2(model_object=clf, test_data=test_df, box_id=test_box_id,
                                                   xy=box_lat_lon, model_type='rand', box_metadata=test_box_meta)

    return results_nearest, results_out, results_majority, results_random


def nn_vs_out_of_the_box_model(all_data=None, test_box_id=1000,
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
    train_df, test_df = prepare_data_for_training_testing(data=all_data, box_id=test_box_id,
                                                          min_test_cases=min_test_cases,
                                                          prop_test=prop_test, max_test_cases=max_test_cases)

    # ----------------CREATE MODEL OBJECT-----------------------------
    num_boxes = len(list(all_data.box_id.unique()))

    etc = ETC(n_estimators=100)
    clf = pred.ImputationNearestNeighbor(data=train_df, target=predictor_params['target'],
                                         neighbors=predictor_params['neighbors'],
                                         how=predictor_params['how'],
                                         time_window=predictor_params['time-window'],
                                         direction=predictor_params['direction'], out_of_box_model=etc,
                                         pred_features=PREDICTION_FEATURES)

    clf.generate_box_metadata(box_file=box_file)
    box_lat_lon = [data_test_bx[data_test_bx.box_id == test_box_id].lat.values[0],
                   data_test_bx[data_test_bx.box_id == test_box_id].lon.values[0]]

    # --------------TEST MODEL PERFOMANCE OUT OF THE BOX---------------------------------------------------
    results_out = compute_metrics_power_state(model_object=clf, test_data=test_df, box_id=test_box_id,
                                              xy=box_lat_lon, model_type='out',)

    # --------------TEST MODEL PERFOMANCE NEAREST NEIGHBOR-----------------------------------------------
    results_nearest = compute_metrics_power_state(model_object=clf, test_data=test_df, box_id=test_box_id,
                                                  xy=box_lat_lon, model_type='nn')

    # --------------TEST MODEL PERFOMANCE MAJORITY CLASSIFIER------------------------------------------
    results_majority = compute_metrics_power_state(model_object=clf, test_data=test_df, box_id=test_box_id,
                                                  xy=box_lat_lon, model_type='major')

    # --------------TEST MODEL PERFOMANCE RANDOM-------------------------------------------------------
    results_random = compute_metrics_power_state(model_object=clf, test_data=test_df, box_id=test_box_id,
                                                   xy=box_lat_lon, model_type='rand')



    return results_nearest, results_out, results_majority, results_random


def batch_evaluation_nn_vs_out_of_the_box_parallel(config_obj=None, model_params=None, eval_params=None,
                                                   file_box_metadata=None):
    """
    use multiprocessing to process all boxes at once.
    :param test_box_id:
    :param model_params:
    :param eval_params:
    :param df_missing:
    :param df_invalid:
    :param df_non_missing:
    :param box_metadata:
    :return:
    """
    test_box_id = 1000

    # --------------PASS WHOLE DATASET OR ONLY DATASET BASED ON NEIGHBORS AND TEST BOX----------#
    df_non_missing, df_missing, df_invalid = get_data_to_evaluate(config_obj=config_obj)

    if model_params['neighbors'] == -1:
        df_bx = df_non_missing
    else:
        # Lets just keep data from the nearest 10 neighbors only
        df_bx = preselect_data_based_on_test_box(df_all=df_non_missing, box_id=test_box_id,
                                                 box_metadata_file=file_box_metadata, k=10)

    # -------------EXTRACT METADATA FOR  THE TEST BOX-------------------------------------------#
    test_box_details = extract_test_box_metadata(config_obj=config, test_box_id=test_box_id, model_params=model_params)

    try:
        # -------------EVALUATE THIS BOX-------------------------------------------------------------#
        results = nn_vs_out_of_the_box_model_parallel(all_data=df_bx, test_box_id=test_box_id,
                                                           prop_test=eval_params['num_tests'],
                                                           min_test_cases=eval_params['min_cases'],
                                                           max_test_cases=eval_params['max_cases'],
                                                           predictor_params=model_params,
                                                          box_file=file_box_metadata, test_box_meta=test_box_details)
        return results
    except Exception as e:
        print_exception()
        return None


def batch_evaluation_nearest_neighbor_vs_out_of_the_box(config_obj=None, model_params=None, eval_params=None):
    """
    This just evaluates the nearest box predictor with a temporal window
    :param df:
    :param use_random:
    :return: a list with accuracy for each PSU
    """
    box_metadata_file = config_obj.get_data_dir() + "Boxes.csv"

    # -----------GET EVALUATION DATA----------------------------------------------#
    df, df_missing, df_invalid = get_data_to_evaluate(config_obj=config_obj)

    # -----------SORT BOXES BY NUMBER OF MISSING EVENTS-----------------------------#
    df_missing_grps = df_missing.groupby(['box_id'])['power_state'].count()
    df_missing_grps = df_missing_grps.reset_index()
    df_missing_grps = df_missing_grps.sort_values(by='power_state', ascending=False)

    num_boxes = eval_params['num_boxes']
    df_missing_grps_top_n = df_missing_grps[:num_boxes]
    boxes = list(df_missing_grps_top_n.box_id.unique())

    # dict object to hold all results
    acc_all = []

    i = 0
    for bx in boxes:
        try:
            # --------------PASS WHOLE DATASET OR ONLY DATASET BASED ON NEIGHBORS AND TEST BOX----------#
            if model_params['neighbors'] == -1:
                df_bx = df
            else:
                df_bx = preselect_data_based_on_test_box(df_all=df, box_id=bx, box_metadata_file=box_metadata_file,
                                                         k=10)

            # --------------PRINTING JUST TO CHECK PROGRESS----------#
            if i % 10 == 0:
                print('{} boxes processed....'.format(i))

            # -------------EVALUATE THIS BOX-------------------------------------------------------------#
            results = nearest_neighbor_vs_out_of_the_box_model(all_data=df_bx, test_box_id=bx,
                                                               prop_test=eval_params['num_tests'],
                                                               min_test_cases=eval_params['min_cases'],
                                                               max_test_cases=eval_params['max_cases'],
                                                               predictor_params=model_params,
                                                               box_file=box_metadata_file)

            # -------------ADD MORE DETAILS TO RESULTS----------------------------------------------------#
            # log number of missing events for each box
            num_missing = df_missing[df_missing.box_id == bx].shape[0]
            num_invalid = df_invalid[df_invalid.box_id == bx].shape[0]
            non_missing = df[df.box_id == bx].shape[0]

            # results contains 2 Metrics objects (one for out of the box model and other for nearest neighbor)
            # add box details to these objects
            bx_df = df_bx[df_bx.box_id == bx]
            for ob in results.__iter__():
                ob.total_non_missing_events = df.shape[0]
                ob.num_invalid_events = num_invalid
                ob.num_missing_events = num_missing
                ob.num_non_missing_events = non_missing
                ob.psu = bx_df.iloc[0].psu
                ob.region = bx_df.iloc[0].region
                ob.window_length = model_params['time-window']
                ob.neighbors = model_params['neighbors']

                acc_all.append(ob)

            i += 1
        except Exception as e:
            print_exception()
            continue

    return acc_all


def experiment_nearest_neighbor_vs_out_of_the_box2(configs=None):
    """
    Simply bundles togather the code for completely evaluating NN
    :return:
    """

    # -----SET UP DATA SOURCES----------------------------------
    output_dir = configs.get_outputs_dir()

    # ----------- EVALUATE IMPUTATION-NEAREST NEIGHBOR----------
    neighbors = [0]
    window = [7]
    window_neighbor_comb = [x for x in itertools.product(window, neighbors)]

    # parameter settings
    params_model = {'neighbors': 1, 'time-window': 7, 'direction': 'both', 'how': 'frequent',
                    'target': 'power_state'}

    num_boxes = 500
    debug_mode = configs.debug_mode
    if debug_mode:
        num_boxes = 10

    params_eval = {'num_tests': 0.20, 'min_cases': 25, 'max_cases': 50,
                   'num_boxes': num_boxes}  # hold out 10 percent of the data

    all_res = []

    # test different combination of parameter
    start = datetime.now()

    results_folder = os.path.join(output_dir, start.strftime('%Y-%m-%d') + '_nn_vs_out_of_box_model')

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    out_file_name = os.path.join(results_folder, 'nn_vs_out_of_box_test.csv')
    dict_from_metrics = []

    # write the features being used
    f = open(os.path.join(results_folder, 'prediction_features.txt'), 'w')
    f.write(','.join(PREDICTION_FEATURES)) # python will convert \n to os.linesep
    f.close()

    for (w, n) in window_neighbor_comb:
        try:
            print()
            print('=====================================================================================')
            print('EVALUATING NEAREST NEIGHBOR & OUT OF THE BOX MODEL ON THE FOLLOWING PARAMETERS:')
            print('Neighbors--> {}, window length-->{} days'.format(n, w))
            print('=====================================================================================')
            start_1 = datetime.now()
            params_model['time-window'] = w
            params_model['neighbors'] = n
            res = batch_evaluation_nn_vs_out_of_the_box_parallel(config_obj=configs, model_params=params_model,
                                                                      eval_params=params_eval)
            all_res = all_res + res
            end_1 = datetime.now()
            time_taken_1 = (end_1 - start_1).total_seconds() / 60
            dict_from_metrics = dict_from_metrics + [m.__dict__ for m in all_res]
            filename = 'w{}n{}_nn_vs_out_of_box.csv'.format(w,n)
            out_file_name_ = os.path.join(results_folder, filename)
            save_metric_objects(out_file=out_file_name_, metrics=[m.__dict__ for m in all_res])
            print('DONE, TOOK {0:.2f} minutes'.format(time_taken_1))

        except Exception as e:
            print_exception()
            # save results for each w,m
            save_metric_objects(out_file=out_file_name, metrics=dict_from_metrics)

    save_metric_objects(out_file=out_file_name, metrics=dict_from_metrics)
    end = datetime.now()
    time_taken = (end - start).total_seconds() / 60
    print()
    print('TOTAL TIME TAKEN FOR ALL PARAMETER OPTIONS: {0:.2f} minutes'.format(time_taken))
    sys.exit()


def experiment_nearest_neighbor_vs_out_of_the_box(configs=None):
    """
    Simply bundles togather the code for completely evaluating NN
    :return:
    """

    # -----SET UP DATA SOURCES----------------------------------
    output_dir = configs.get_outputs_dir()

    # ----------- EVALUATE IMPUTATION-NEAREST NEIGHBOR----------
    neighbors = [-1, 0, 1, 10]
    window = [-1, 7, 15, 30]
    window_neighbor_comb = [x for x in itertools.product(window, neighbors)]

    # parameter settings
    params_model = {'neighbors': 1, 'time-window': 7, 'direction': 'both', 'how': 'frequent',
                    'target': 'power_state'}

    num_boxes = 500
    debug_mode = configs.debug_mode
    if debug_mode:
        num_boxes = 25

    params_eval = {'num_tests': 0.20, 'min_cases': 25, 'max_cases': 50,
                   'num_boxes': num_boxes}  # hold out 10 percent of the data

    all_res = []

    # test different combination of parameter
    start = datetime.now()

    results_folder = os.path.join(output_dir, start.strftime('%Y-%m-%d') + '_nn_vs_out_of_box_model')

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    out_file_name = os.path.join(results_folder, 'nn_vs_out_of_box.csv')
    dict_from_metrics = []

    # write the features being used
    f = open(os.path.join(results_folder, 'prediction_features.txt'), 'w')
    f.write(','.join(PREDICTION_FEATURES)) # python will convert \n to os.linesep
    f.close()

    for (w, n) in window_neighbor_comb:
        try:
            print()
            print('=====================================================================================')
            print('EVALUATING NEAREST NEIGHBOR & OUT OF THE BOX MODEL ON THE FOLLOWING PARAMETERS:')
            print('Neighbors--> {}, window length-->{} days'.format(n, w))
            print('=====================================================================================')
            start_1 = datetime.now()
            params_model['time-window'] = w
            params_model['neighbors'] = n
            res = batch_evaluation_nearest_neighbor_vs_out_of_the_box(config_obj=configs, model_params=params_model,
                                                                      eval_params=params_eval)
            all_res = all_res + res
            end_1 = datetime.now()
            time_taken_1 = (end_1 - start_1).total_seconds() / 60
            dict_from_metrics = dict_from_metrics + [m.__dict__ for m in all_res]
            filename = 'w{}n{}_nn_vs_out_of_box.csv'.format(w,n)
            out_file_name_ = os.path.join(results_folder, filename)
            save_metric_objects(out_file=out_file_name_, metrics=[m.__dict__ for m in all_res])
            print('DONE, TOOK {0:.2f} minutes'.format(time_taken_1))

        except Exception as e:
            print_exception()
            # save results for each w,m
            save_metric_objects(out_file=out_file_name, metrics=dict_from_metrics)

    save_metric_objects(out_file=out_file_name, metrics=dict_from_metrics)
    end = datetime.now()
    time_taken = (end - start).total_seconds() / 60
    print()
    print('TOTAL TIME TAKEN FOR ALL PARAMETER OPTIONS: {0:.2f} minutes'.format(time_taken))
    sys.exit()


def run_experiment(experiment_name=None):
    """
    Runs a specific experiment
    :param experiment_name:
    :return:
    """

    if experiment_name=='experiment_nearest_neighbor_vs_out_of_the_box2':
        experiment_nearest_neighbor_vs_out_of_the_box2(configs=config)
    elif experiment_name=='experiment_nearest_neighbor_vs_out_of_the_box':
        experiment_nearest_neighbor_vs_out_of_the_box(configs=config)


if __name__ == "__main__":
    # ----------SET UP-----------------------------------------
    config = prep.Configurations()
    config.debug_mode = True

    # -----------NEAREST NEIGHBOR VS. OUT OF THE BOX----------------------
    exp_name = 'experiment_nearest_neighbor_vs_out_of_the_box2'
    run_experiment(experiment_name=exp_name)
