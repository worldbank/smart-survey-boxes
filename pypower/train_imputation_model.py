
"""
Trains ETC model once in 2 weeks!!!!
"""
import multiprocessing

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.externals import joblib
import pickle

from pypower import preprocessing as prep


def train_and_save_etc_model(config_obj=None, target='event_type_num', debug=False):
    """
    Returns sms_rect_hr file with missing values filled
    :return:
    """

    # -------------READ IN DATA--------------
    file_sms_rect_hr = config_obj.get_processed_data_dir() + 'sms_rect_hr.csv'
    model_dir = config_obj.get_model_dir()
    model_name = model_dir + 'ETC.pkl'

    cols_to_use = ['box_id', 'psu', 'lon', 'lat', 'str_datetime_sent_hr', 'day_sent', 'hour_sent', 'month_sent',
                   'wk_day_sent', 'wk_end', 'event_type_num', 'event_type_str', 'data_source']

    df = pd.read_csv(file_sms_rect_hr, usecols=cols_to_use)

    if debug:
        df = df.sample(n=100000)

    # drop missing and test events
    num_missing = len(df[df.event_type_str == 'missing'])
    print('Number of missing events...{} out of total {} in rectangular dataset (sms_rect_hr.csv)'.format(num_missing, df.shape[0]))
    print('Discard missing events...we dont need them for training...training the model, this could take a while....')

    df = df[df.event_type_str != 'missing']

    # -------------FIX MODEL PARAMETERS----------------------------------
    # These parameters and features and features give best perfomance as of now
    predictor_params = {'n_estimators': 200, 'n_jobs': -1, 'criterion': 'gini'}

    prediction_features = ['box_id', 'psu', 'lon', 'lat', 'hour_sent', 'month_sent', 'day_sent', 'wk_day_sent',
                           'wk_end']

    X = df[prediction_features].values
    y = df[target].values
    clf = ExtraTreesClassifier(**predictor_params)
    clf.fit(X, y)

    # -------------PICKLE MODEL---------------
    joblib.dump(clf, model_name)


def train_and_save_gbm_model(config_obj=None, target='power_state', debug=False):
    """
    Returns sms_rect_hr file with missing values filled
    :return:
    """

    # -------------READ IN DATA--------------
    file_sms_rect_hr = config_obj.get_processed_data_dir() + 'sms_rect_hr.csv'
    model_dir = config_obj.get_model_dir()
    model_name = model_dir + 'gbm.pkl'

    cols_to_use = ['box_id', 'psu', 'lon', 'lat', 'str_datetime_sent_hr', 'day_sent', 'hour_sent', 'month_sent',
                   'wk_day_sent', 'wk_end', 'event_type_num', 'event_type_str', 'power_state', 'data_source']

    df = pd.read_csv(file_sms_rect_hr, usecols=cols_to_use)

    if debug:
        df = df.sample(n=100000)

    # drop missing and test events
    num_missing = len(df[df.power_state == -1])
    print('Number of missing events...{} out of total {} in rectangular dataset (sms_rect_hr.csv)'.format(num_missing, df.shape[0]))
    print('Discard missing events...we dont need them for training...training the model, this could take a while....')

    df = df[df.power_state != -1]

    # -------------FIX MODEL PARAMETERS----------------------------------
    # These parameters and features and features give best perfomance as of now
    predictor_params = {'n_estimators': 100, 'random_state':1}

    prediction_features = ['box_id', 'psu', 'lon', 'lat', 'hour_sent', 'month_sent', 'day_sent', 'wk_day_sent',
                           'wk_end']

    X = df[prediction_features].values
    y = df[target].values
    clf = GradientBoostingClassifier(**predictor_params)
    clf.fit(X, y)

    # -------------PICKLE MODEL---------------
    joblib.dump(clf, model_name)
    print('Done with model training and save it to file')

if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver', force=True)

    # ------------SET UP WORKING DIRECTORY AND FILES--------------------
    # create config object and set imputation approach to nearest neighbor
    config = prep.Configurations(platform='mac', imputation_approach='etc', debug_mode=False)

    # -----------TRAIN IMPUTATION MODEL----------------------------------------
    # train_and_save_etc_model(config,debug=config.debug_mode)
    train_and_save_gbm_model(config, debug=config.debug_mode)

