"""
This is where we do the actual data preprocessing including imputation
"""

import sys
import os
import traceback
from datetime import datetime

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from pypower import data_utils as ut
from pypower import prediction_models as pred


class Configurations:
    """
    Utility class mainly for resource location.
    """

    DATE_FORMAT = '%Y-%m-%d'

    PLATFORM = 'mac'

    # windows-onedrive
    WINDOWS_DATA_DIR = "C:/Users/wb344850/WBG/William Hutchins Seitz - 01.data/"
    WINDOWS_XML_DIR = "C:/Users/wb344850/Google Drive/SMSBuckupRestore/"
    WINDOWS_RAW_SMS_DIR = "C:/Users/wb344850/WBG/William Hutchins Seitz - 01.data/raw_sms/"
    WINDOWS_PROCESSED_SMS_DIR = "C:/Users/wb344850/WBG/William Hutchins Seitz - 01.data/processed_sms/"
    WINDOWS_MODEL_DIR = "C:/Users/wb344850/WBG/William Hutchins Seitz - 01.data/imputation_models/"
    WINDOWS_OUTPUT_DIR = ''

    # mac
    MAC_DATA_DIR = "/Users/dmatekenya/Google Drive/World-Bank/electricity_monitoring/01.data/"
    MAC_XML_DIR = "/Users/dmatekenya/Google Drive/SMSBuckupRestore/"
    MAC_RAW_SMS_DIR = "/Users/dmatekenya/Google Drive/World-Bank/electricity_monitoring/01.data/raw_sms/"
    MAC_PROCESSED_SMS_DIR = "/Users/dmatekenya/Google Drive/World-Bank/electricity_monitoring/01.data/processed_sms/"
    MAC_MODEL_DIR = "/Users/dmatekenya/imputation_models/"
    MAC_OUTPUTS_DIR = "/Users/dmatekenya/PycharmProjects/power-mon/outputs/"

    def __init__(self,platform='mac', imputation_approach='etc', debug_mode=True):
        self.platform = platform
        self.data_dir = None
        self.output_dir = None
        self.imputation_approach = imputation_approach
        self.debug_mode = debug_mode

    def get_xml_dir(self):
        if self.platform == 'mac':
            return self.MAC_XML_DIR
        elif self.platform == 'bank_windows':
            path = os.path.normpath(self.WINDOWS_XML_DIR) + '\\'
            return path

    def get_data_dir(self):
        if self.platform == 'mac':
            return self.MAC_DATA_DIR
        elif self.platform == 'bank_windows':
            path = os.path.normpath(self.WINDOWS_DATA_DIR) + '\\'
            return path

    def get_raw_data_dir(self):
        if self.platform == 'mac':
            return self.MAC_RAW_SMS_DIR
        elif self.platform == 'bank_windows':
            path = os.path.normpath(self.WINDOWS_RAW_SMS_DIR) + '\\'
            return path

    def get_processed_data_dir(self):
        if self.platform == 'mac':
            return self.MAC_PROCESSED_SMS_DIR
        elif self.platform == 'bank_windows':
            path = os.path.normpath(self.WINDOWS_PROCESSED_SMS_DIR) + '\\'
            return path

    def get_outputs_dir(self):
        if self.platform == 'mac':
            return self.MAC_OUTPUTS_DIR
        elif self.platform == 'bank_windows':
            path = os.path.normpath(self.MAC_OUTPUTS_DIR) + '\\'
            return path

    def get_model_dir(self):
        if self.platform == 'mac':
            return self.MAC_MODEL_DIR
        elif self.platform == 'bank_windows':
            path = os.path.normpath(self.WINDOWS_MODEL_DIR) + '\\'
            return path


def impute_with_universal_model(config_obj='config',how='nn'):
    """
    Fills in missing values using either nearest neighbor(nn) or an out of box model such as random forest
    and saves result as 'sms_v3.csv'.
    Here, we use same parameters and model for all boxes.
    :param params:
    :return:
    """
    box_file = config_obj.get_data_dir() + 'Boxes.csv'
    file_sms_rect_hr = config_obj.get_processed_data_dir() + 'sms_rect_hr.csv'  # file with missing values
    file_sms_v3 = config_obj.get_processed_data_dir() + 'sms_v3.csv'  # file after filling missing values

    if how == 'nn':
        # --------impute with nearest neighbor--------------------
        nn_params = {'neighbors': 1, 'time-window': 7, 'direction': 'both', 'how': 'frequent'}
        df_imputed = impute_with_nearest_neighbor(file_sms_rect_hr=file_sms_rect_hr, predictor_params=nn_params, bx_file=box_file)

        df_imputed.to_csv(file_sms_v3, index=False) # save to file
    elif how == 'out':
        # --------impute with out of box model--------------------
        model = 'RF'
        prediction_features = ['box_id', 'psu', 'lon', 'lat', 'month_sent', 'wk_day_sent', 'wk_end', 'holiday']
        params = {'trees': 1, 'pred_feat': prediction_features}
        df_imputed = impute_with_out_of_box_model(file_sms_rect_hr=file_sms_rect_hr, predictor_params=params,
                                                  model_name=model, target_var='event_type_num')
        df_imputed.to_csv(file_sms_v3, index=False)  # save to file


def impute_with_out_of_box_model(file_sms_rect_hr=None, predictor_params=None,target_var='event_type_num', model_name='LR'):
    """
    :param file_sms_rect_hr: version 2 of sms data
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

    df = pd.read_csv(file_sms_rect_hr, usecols=list(cols_to_use), nrows=10000)

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


def impute_with_nearest_neighbor(file_sms_rect_hr=None, predictor_params = None, bx_file =None):
    """
    Returns sms_rect_hr file with missing values filled
    :return:
    """

    # -------------READ IN DATA--------------
    cols_to_use = ['box_id', 'psu', 'lon', 'lat', 'str_datetime_sent','data_source',
                   'str_datetime_sent_hr', 'hour_sent', 'event_type_str','event_type_num']

    df = pd.read_csv(file_sms_rect_hr, usecols=cols_to_use, parse_dates=['str_datetime_sent_hr'])

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


def replace_event_type_str(num):
    """
    Given event_type_num return the string.
    :param num:
    :return:
    """
    event_type = {1: 'pback', 2: 'pfail', 3: 'pon_mon', 4: 'pfail_mon'}
    return event_type.get(num)


def impute_with_etc(config_obj=None, prediction_features=None):
    """
    Fill out missing events using scikit-learn Extra Trees Classifier based.
    :param config_obj:
    :param prediction_features:
    :return:
    """
    file_sms_rect_hr = config_obj.get_processed_data_dir() + 'sms_rect_hr.csv'

    model_file = config_obj.get_model_dir() + 'ETC.pkl'

    file_sms_rect_hr_imp = config_obj.get_processed_data_dir() + 'sms_rect_hr_imp.csv'

    df = pd.read_csv(file_sms_rect_hr)

    df_missing = df[df['event_type_num'] == -1]

    if not prediction_features:
        prediction_features = ['box_id', 'psu', 'lon', 'lat', 'hour_sent', 'month_sent', 'day_sent', 'wk_day_sent',
                           'wk_end']

    X = df_missing[prediction_features]

    try:
        clf = joblib.load(model_file)

        y_predicted = clf.predict(X)

        df.loc[df['event_type_num'] == -1, 'event_type_num'] = y_predicted

        # replace missing event_type_str
        df.loc[df['event_type_str'] == 'missing', 'event_type_str'] = df['event_type_num'].apply(
            lambda x: replace_event_type_str(x))

        df.to_csv(file_sms_rect_hr_imp, index=False)

    except Exception as e:
        print(e)

    print('SUCCESSFULLY IMPUTED WITH ETC')


def preprocesss_raw_sms(configuration=None, debugging=True):
    """
    Takes raw sms.xml and converts it into sms_observed (observed_events) and sms_rect_hr (observed + inserted events)
    :param configuration: Has details about file locations
    :param debugging: whether to run in debug mode or not
    :return:
    """
    try:
        box_file = configuration.get_data_dir() + 'Boxes.csv'
        xml_file = configuration.get_xml_dir() + 'sms.xml'
        sms_observed = configuration.get_processed_data_dir() + 'sms_observed.csv'  # filename for sms_observed based on date
        sms_rect_hr = configuration.get_processed_data_dir() + 'sms_rect_hr.csv'  # filename for sms_rect_hr based on date

        start = datetime.now()
        ut.process_raw_sms(sms_observed_file=sms_observed, sms_rect_hr_file=sms_rect_hr, raw_sms=xml_file,
                           box_details=box_file,debug_mode=debugging)
        end = datetime.now()
        print('Processing took {} seconds '.format((end - start).total_seconds()))
    except Exception as e:
        desired_trace = traceback.format_exc(sys.exc_info())
        print(desired_trace)
