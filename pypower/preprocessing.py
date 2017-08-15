'''
This is where we do the actual data preprocessing including imputation
'''

import os, sys
import traceback
import multiprocessing
from datetime import datetime
import time
import pandas as pd
import numpy as np
from pypower import utils as ut
from pypower import model_selection as mod_ev
from pypower import prediction_models as pred
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


class Configurations:
    """
    Utility class mainly for resource location.
    """

    DATE_FORMAT = '%Y-%m-%d'

    PLATFORM = 'mac'

    # windows-onedrive
    WINDOWS_DATA_DIR = "C:/Users/wb344850/WBG/William Hutchins Seitz - 01.data/"
    WINDOWS_RAW_SMS_DIR = "C:/Users/wb344850/WBG/William Hutchins Seitz - 01.data/raw_sms/"
    WINDOWS_PROCESSED_SMS_DIR = "C:/Users/wb344850/WBG/William Hutchins Seitz - 01.data/processed_sms/"
    WINDOWS_MODEL_DIR = "C:/Users/wb344850/WBG/William Hutchins Seitz - 01.data/imputation_models/"
    WINDOWS_OUTPUT_DIR = ''

    # mac
    MAC_DATA_DIR = "/Users/dmatekenya/Google Drive/World-Bank/electricity_monitoring/01.data/"
    MAC_RAW_SMS_DIR = "/Users/dmatekenya/Google Drive/World-Bank/electricity_monitoring/01.data/raw_sms/"
    MAC_PROCESSED_SMS_DIR = "/Users/dmatekenya/Google Drive/World-Bank/electricity_monitoring/01.data/processed_sms/"
    MAC_OUTPUTS_DIR = "/Users/dmatekenya/PycharmProjects/power-mon/outputs/"

    def __init__(self,platform='mac', imputation_approach = 'nn', debug_mode=True):
        self.platform = platform
        self.data_dir = None
        self.output_dir = None
        self.imputation_approach = imputation_approach
        self.debug_mode = debug_mode

    def get_data_dir(self):
        if self.platform == 'mac':
            return self.MAC_DATA_DIR
        elif self.platform == 'bank_windows':
            return self.WINDOWS_DATA_DIR

    def get_raw_data_dir(self):
        if self.platform == 'mac':
            return self.MAC_RAW_SMS_DIR
        elif self.platform == 'bank_windows':
            return self.WINDOWS_RAW_SMS_DIR

    def get_processed_data_dir(self):
        if self.platform == 'mac':
            return self.MAC_PROCESSED_SMS_DIR
        elif self.platform == 'bank_windows':
            return self.WINDOWS_PROCESSED_SMS_DIR

    def get_outputs_dir(self):
        if self.platform == 'mac':
            return self.MAC_OUTPUTS_DIR
        elif self.platform == 'bank_windows':
            return self.WINDOWS_OUTPUT_DIR


def impute_with_universal_model(config_obj='config',how='nn'):
    """
    Fills in missing values using either nearest neighbor(nn) or an out of box model such as random forest
    and saves result as 'sms_v3.csv'.
    Here, we use same parameters and model for all boxes.
    :param params:
    :return:
    """
    box_file = config_obj.get_data_dir() + 'Boxes.csv'
    file_sms_v2 = config_obj.get_processed_data_dir() + 'sms_v2.csv'  # file with missing values
    file_sms_v3 = config_obj.get_processed_data_dir() + 'sms_v3.csv'  # file after filling missing values

    if how == 'nn':
        # --------impute with nearest neighbor--------------------
        nn_params = {'neighbors': 1, 'time-window': 7, 'direction': 'both', 'how': 'frequent'}
        df_imputed = impute_with_nearest_neighbor(file_sms_v2=file_sms_v2, predictor_params=nn_params, bx_file=box_file)

        df_imputed.to_csv(file_sms_v3, index=False) # save to file
    elif how == 'out':
        # --------impute with out of box model--------------------
        model = 'RF'
        prediction_features = ['box_id', 'psu', 'lon', 'lat', 'month_sent', 'wk_day_sent', 'wk_end', 'holiday']
        params = {'trees': 1, 'pred_feat': prediction_features}
        df_imputed = impute_with_out_of_box_model(file_sms_v2=file_sms_v2, predictor_params=params,
                                                  model_name=model, target_var='event_type_num')
        df_imputed.to_csv(file_sms_v3, index=False)  # save to file


def impute_with_out_of_box_model(file_sms_v2=None, predictor_params=None,target_var='event_type_num', model_name='LR'):
    """
    :param file_sms_v2: version 2 of sms data
    :param predictor_params: Parameters for predictors-to be sourced dynamically from file
    :param bx_file: Box metadata
    :return: A Pandas dataframe with missing data filled.
    """
    # Potential models
    random_state = 1
    clfs = {'LR': LogisticRegression(penalty='l1', max_iter=50),
            'RF': RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                         random_state=random_state),
            'GBM': GradientBoostingClassifier(n_estimators=500,
                                              random_state=random_state),
            'ETC': ExtraTreesClassifier(n_estimators=500, n_jobs=-1, criterion='gini')
            }

    clf = clfs[model_name]
    pred_feat = predictor_params['pred_feat']
    # -------------READ IN DATA----------------------------------------------------
    cols_to_use = set(['box_id', 'psu', 'lon', 'lat', 'datetime_sent_raw', 'str_datetime_sent',
                   'str_datetime_sent_hr', 'hour_sent', 'event_type_str', 'event_type_num',
                   'power_state', 'data_source'] + pred_feat)

    df = pd.read_csv(file_sms_v2, usecols=list(cols_to_use), nrows=10000)

    train_df = df[df.event_type_num != -1]  # Keep only non-missing values for evaluation
    df_missing = df[df.event_type_num == -1]
    df_missing.is_copy = False
    print('{} missing values to be computed out of {} total events'.format(df_missing.shape[0], df.shape[0]))

    # -------------TRAIN MODEL------------------------------------------------------
    X = train_df[pred_feat].values
    y = train_df[target_var].values

    clf.fit(X=X, y=y)

    # ------------IMPUTE------------------------------------------------------------
    X_predict = df_missing[pred_feat].values

    y_predicted = clf.predict(X_predict)

    df_missing['event_type_num'] = pd.Series(y_predicted)

    # ------------COMBINE THE TWO DF's and ENSURE NOMORE MISSING--------------------
    df = train_df.append(df_missing)
    num_missing = df[df.event_type_num == -1].shape[0]

    if num_missing == 0:
        print('ALL MISSING VALUES FILLED USING {} '.format(model_name))

    return df


def impute_with_nearest_neighbor(file_sms_v2=None, predictor_params = None, bx_file =None):
    """
    Returns sms_v2 file with missing values filled
    :return:
    """

    # -------------READ IN DATA--------------
    cols_to_use = ['box_id', 'psu', 'lon', 'lat', 'str_datetime_sent','data_source',
                   'str_datetime_sent_hr', 'hour_sent', 'event_type_str','event_type_num']

    df = pd.read_csv(file_sms_v2, usecols=cols_to_use, parse_dates=['str_datetime_sent_hr'])

    df.rename(columns={'str_datetime_sent_hr': 'datetime_sent_hr'}, inplace=True)
    train_df = df[df.event_type_str != 'missing']  # Keep only non-missing values for evaluation
    df_missing = df[df.event_type_str == 'missing']
    df_missing.is_copy = False
    print('{} missing values to be computed out of {} total events'.format(df_missing.shape[0], df.shape[0]))

    # -------------CREATE MODEL---------------
    clf = pred.ImputationNearestNeighbor(data=train_df,
                                         neighbors=predictor_params['neighbors'],
                                         how=predictor_params['how'],
                                         time_window=predictor_params['time-window'],
                                         direction=predictor_params['direction']
                                         )

    clf.generate_box_metadata(box_file=bx_file)

    # ------------IMPUTE------------------------------------------------------------
    df_missing['event_type_str'] = df_missing.apply (
        lambda x: clf.predict(prediction_date=x['datetime_sent_hr'], box_id=x['box_id'],
                              target_loc=[x['lat'],x['lon']]), axis=1)

    # ------------COMBINE THE TWO DF's and ENSURE NOMORE MISSING--------------------
    df = train_df.append(df_missing)
    num_missing = df[df.event_type_str == 'missing'].shape[0]

    if num_missing == 0:
        print('ALL MISSING VALUES FILLED USING NEAREST NEIGHBOR MODEL')

    return df


def preprocesss_raw_sms(configuration=None, debugging=True):
    """
    Takes raw sms.xml and converts it into sms_v1 (observed_events) and sms_v2 (observed + inserted events)
    :param configuration: Has details about file locations
    :param debugging: whether to run in debug mode or not
    :return:
    """
    try:
        box_file = configuration.get_data_dir() + 'Boxes.csv'
        xml_file = configuration.get_data_dir() + 'sms.xml'
        sms_v1 = configuration.get_processed_data_dir() + 'sms_v1.csv'  # filename for sms_v1 based on date
        sms_v2 = configuration.get_processed_data_dir() + 'sms_v2.csv'  # filename for sms_v2 based on date

        start = datetime.now()
        ut.process_raw_sms(sms_v1_file=sms_v1, sms_v2_file=sms_v2, raw_sms=xml_file, box_details=box_file,
                           debug_mode=debugging)
        end = datetime.now()
        print('Processing took {} seconds '.format((end - start).total_seconds()))
    except Exception as e:
        desired_trace = traceback.format_exc(sys.exc_info())
        print(desired_trace)


def main(impute=False, preprocess=True, config_object=None):

    if impute:
        # --------IMPUTATIONS------------------------------------------------
        # TODO assign models specific to each box after doing comprehensive evaluation
        how_to_impute = config_object.imputation_approach

        # currently using one model for all boxes.....
        impute_with_universal_model(config_obj=config_object, how=how_to_impute)
    if preprocess:
        # ------------PREPROCESS DATA----------------------------------------
        # Takes raw sms.xml and saves to sms_v1 (only observed events) and sms_v2 (one hour time resolution dataset)
        debug = config.debug_mode
        preprocesss_raw_sms(config, debugging=debug)


if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver', force=True)

    # ------------SET UP WORKING DIRECTORY AND FILES--------------------
    # create config object and set imputation approach to nearest neighbor
    config = Configurations(platform='mac', imputation_approach='nn', debug_mode=False)

    # -----------PREPROCESS ONLY----------------------------------------
    main(preprocess=True, impute=False, config_object=config)





