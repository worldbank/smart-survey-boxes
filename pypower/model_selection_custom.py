'''
This module is for evaluating model perfomance
'''

import itertools
import multiprocessing
import random
import sys
import traceback
from datetime import datetime
from collections import namedtuple

import numpy as np
import pandas as pd

from pypower import data_utils as ut
from pypower import prediction_models as pred
from pypower import preprocessing as prep


def batch_evaluation_nearest_neighbor(config_obj=None, model_params=None, eval_params=None):
    """
    This just evaluates the nearest box predictor with a temporal window
    :param df:
    :param use_random:
    :return: a list with accuracy for each PSU
    """

    # Files
    file_sms_v2 = config_obj.get_processed_data_dir() + "sms_rect_hr.csv"
    box_metadata_file = config_obj.get_data_dir() + "Boxes.csv"

    # read in the data
    cols_to_use = ['box_id', 'psu', 'lon', 'lat', 'str_datetime_sent_hr', 'hour_sent', 'event_type_str', 'power_state']

    df = pd.read_csv(file_sms_v2, usecols=cols_to_use, parse_dates=['str_datetime_sent_hr'])

    df.rename(columns={'str_datetime_sent_hr': 'datetime_sent_hr'}, inplace=True)

    df = df[df.event_type_str != 'missing']  # Keep only non-missing values for evaluation

    # list of all boxes
    box_list = list(df.box_id.unique())
    if eval_params['num_boxes'] > len(box_list):
        boxes = box_list
    else:
        boxes = random.sample(box_list, eval_params['num_boxes'])  # select only a few boxes

    # dict object to hold all results
    acc_all = {}

    i = 0
    for bx in boxes:
        try:
            # preselect data based on box location
            bx_df = df[df.box_id == bx]

            box_xy = [bx_df['lat'].iloc[0], bx_df['lon'].iloc[0]]
            neighbors = ut.k_nearest_boxes(box_metadata_file=box_metadata_file, box_id=bx, box_lat_lon=box_xy, k=10)
            neighbors.append(bx)

            df_bx = df[df['box_id'].isin(neighbors)]

            if i % 10 == 0:
                print('{} boxes processed....'.format(i))

            results = evaluate_nearest_neighbor_model(all_data=df_bx, test_box_id=bx,
                                                      prop_test=eval_params['num_tests'],
                                                      min_test_cases=eval_params['min_cases'],
                                                      max_test_cases=eval_params['max_cases'],
                                                      predictor_params=model_params,
                                                      box_file=box_metadata_file)

            acc_all[bx] = results
            i += 1
        except Exception as e:
            desired_trace = traceback.format_exc(sys.exc_info())
            print(desired_trace)
            continue

    return acc_all


def evaluate_nearest_neighbor_model(all_data=None, test_box_id=1000,
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
        return None

    if num_tests > max_test_cases:
        num_tests = max_test_cases

    test_dates = random.sample(event_dates, num_tests)

    to_keep = list(set(event_dates) - set(test_dates))

    train_df = data_test_bx[data_test_bx['datetime_sent_hr'].isin(to_keep)]

    test_df = data_test_bx[data_test_bx['datetime_sent_hr'].isin(test_dates)]

    # ----------------CREATE MODEL OBJECT-----------------------------
    clf = pred.ImputationNearestNeighbor(data=train_df, target=predictor_params['target'],
                                         neighbors=predictor_params['neighbors'],
                                         how=predictor_params['how'],
                                         time_window=predictor_params['time-window'],
                                         direction=predictor_params['direction']
                                         )

    clf.generate_box_metadata(box_file=box_file)
    box_lat_lon = [data_test_bx[data_test_bx.box_id == test_box_id].lat.values[0],
                   data_test_bx[data_test_bx.box_id == test_box_id].lon.values[0]]

    # --------------TEST MODEL PERFOMANCE---------------------------
    results = nearest_neigbor_model_metrics_power_state(model_object=clf, test_data=test_df, box_id=test_box_id,
                                                        xy=box_lat_lon)

    return results


def nearest_neigbor_model_metrics_power_state(model_object=None, test_data=None, box_id=90, xy=None,
                                              target='power_state'):
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
        predicted = model_object.predict(prediction_date=test_date, box_id=box_id, target_loc=xy)

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
    precision_0, precision_1, recall_0, precision_1 = [np.nan for _ in range(4)]

    if tot_pred_1 > 0:
        precision_1 = correct_1/tot_pred_1 * 100

    if actual_1 > 0:
        recall_1 = correct_1/actual_1 * 100

    if tot_pred_0 > 0:
        precision_0 = correct_0 / tot_pred_0 * 100

    if actual_0 > 0:
        recall_0 = correct_0 / actual_0 * 100

    fieldnames = 'support actual_1 actual_0 accuracy precision_1 recall_1 precision_0 recall_0 tot_pred_0 ' \
                 'tot_pred_1 correct_1 correct_0'

    Metric = namedtuple('Metric', field_names=fieldnames)


    metrics = Metric(actual_1=actual_1, actual_0 = actual_0, accuracy = accuracy, tot_pred_0 = tot_pred_0,
                     tot_pred_1 = tot_pred_1, precision_1 = precision_1,  precision_0 = precision_0,
                     recall_1= recall_1, recall_0=recall_0, support = tot, correct_1 = correct_1, correct_0 = correct_0)

    return metrics


def param_selection_imputation_nearest_neigbor_model(results=None, output_file=None):
    """
    Summarises results and
    :param results:
    :return:
    """
    # -----------CREATE DATAFRAME FROM RESULTS-------------------------------------
    row_list = []

    for k, v in results.items():
        meta_data = {'window_len': k[0], 'neighbors': k[1]}
        try:
            for b_id, metrics in v.items():
                if not metrics:
                    continue
                data_dict = dict()
                data_dict['box_id'] = b_id
                data_dict.update(metrics._asdict())
                data_dict.update(meta_data)

                row_list.append(data_dict)
        except Exception as e:
            print(e)

    # -----------SAVE RESULTS FOR ANALYSIS AND MODEL SELECTION----------------------------------------
    df = pd.DataFrame(row_list)
    df.to_csv(output_file, index=False)
    print('Successfully saved param settings.....')
    return df


def experiment_nearest_neighbor_neighbors_window(configs=None):
    """
    Simply bundles togather the code for completely evaluating NN
    :return:
    """

    # -----SET UP DATA SOURCES----------------------------------
    output_dir = configs.get_outputs_dir()

    # ----------- EVALUATE IMPUTATION-NEAREST NEIGHBOR----------
    neighbors = [0, 1, 2]
    window = [7, 14]
    window_neighbor_comb = [x for x in itertools.product(window, neighbors)]

    # parameter settings
    params_model = {'neighbors': 1, 'time-window': 7, 'direction': 'both', 'how': 'frequent',
                    'target': 'power_state'}

    num_boxes = 500
    debug_mode = configs.debug_mode
    if debug_mode:
        num_boxes = 2

    params_eval = {'num_tests': 0.20, 'min_cases': 50, 'max_cases': 250,
                   'num_boxes': num_boxes}  # hold out 10 percent of the data

    all_res = {}

    # test different combination of parameter
    start = datetime.now()
    print('DOING PARAMETER OPTIMISATION FOR NEAREST NEIGHBOR MODEL....')
    for (w, n) in window_neighbor_comb:
        print('Neighbors--> {}, window length-->{} days'.format(n, w))
        params_model['time-window'] = w
        params_model['neighbors'] = n
        res = batch_evaluation_nearest_neighbor(config_obj=configs, model_params=params_model, eval_params=params_eval)
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

    # -----------EVALUATE NEAREST NEIGHBOR----------------------
    experiment_nearest_neighbor_neighbors_window(configs=config)
