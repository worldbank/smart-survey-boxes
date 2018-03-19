"""
All the data processing is done in this module including:
1. Daily backup of the sms
2. Pre-processing the raw sms and saving them to files
2. Imputing missing data values
"""


import os
import sys
from datetime import datetime


import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# A quick and dirty solution to deal with running the script in command line and using task scheduler
user_id = 'WB255520'
path = None
if user_id == 'WB454594':
    path = r'C:\Users\{}\Google-Drive\smart-survey-boxes\code'.format(user_id)
elif user_id == 'WB255520':
    path = r'C:\Users\{}\Google-Drive\smart-survey-boxes\code'.format(user_id)

sys.path.append(path)

from data_processing import data_processing_utils as ut
from data_processing import prediction_models as pred

# Environment variables: this will need to be set manually
ENV_VARS = {'data_folder': None, 'xml_folder': None, 'outputs_folder': None, 'box_dist_ver': 14}
if sys.platform == 'darwin':
    ENV_VARS['data_folder'] = os.path.abspath('/Users/dmatekenya/Google-Drive/worldbank/smart-survey-boxes/data')
    ENV_VARS['xml_folder'] = os.path.abspath("/Users/dmatekenya/Google-Drive/SMSBuckupRestore/")
    ENV_VARS['outputs_folder'] = os.path.abspath('/Users/dmatekenya/Google-Drive/worldbank/smart-survey-boxes/outputs')


if sys.platform == 'win32':
    if user_id == 'WB454594':
        # windows-onedrive
        ENV_VARS['data_folder'] = os.path.join('C:/', 'Users', user_id, 'OneDrive - WBG','Tajikistan\Listening2Tajikistan\01.Eletricity_monitoring','01.data')
        ENV_VARS['xml_folder'] = os.path.join('C:/', 'Users', user_id, 'Google-Drive', 'SMSBuckupRestore')
        ENV_VARS['outputs_folder'] = os.path.join('C:/', 'Users', user_id, 'OneDrive - WBG','Tajikistan\Listening2Tajikistan\01.Eletricity_monitoring','outputs_from_dunstan_data_processing')
    elif user_id == 'WB255520':
        # windows-onedrive
        ENV_VARS['data_folder'] = os.path.join('C:/', 'Users', user_id, 'temp')
        ENV_VARS['xml_folder'] = os.path.join('C:/', 'Users', user_id, 'Google-Drive', 'SMSBuckupRestore')
        ENV_VARS['outputs_folder'] = os.path.join('C:/', 'Users', user_id, 'temp', 'outputs_from_dunstan_data_processing')
    else:
        # windows-onedrive
        ENV_VARS['data_folder'] = os.path.join('C:/', 'Users', user_id,
                                               'WBG\William Hutchins Seitz - 01.Eletricity_monitoring', '01.data')
        ENV_VARS['xml_folder'] = os.path.join('C:/', 'Users', 'wb344850', 'Google-Drive', 'SMSBuckupRestore')
        ENV_VARS['outputs_folder'] = os.path.join('C:/', 'Users', 'wb344850',
                                                  'WBG\William Hutchins Seitz - 01.Eletricity_monitoring',
                                                  'outputs_from_dunstan_data_processing')


class DataProcessor:
    """
    Utility class for handling data processing
    """
    # number of boxes-currently we get this from Distribution_Boxes@14.xlsx
    NUM_WORKING_BOXES = 298

    def __init__(self, process_type='impute', data_dir=None, xml_dir=None, outputs_dir=None, debug_mode=True,
                 verification_mode=False, box_dist_ver=None):
        self.process_type = process_type
        self.data_dir = data_dir
        self.xml_dir = xml_dir
        self.outputs_dir = outputs_dir
        self.debug_mode = debug_mode
        self.verification_mode = verification_mode
        self.main_data_dir = data_dir
        self.raw_data_dir = os.path.join(self.data_dir, 'raw-sms-backup')
        self.model_dir = os.path.join(self.data_dir, 'imputation-models')
        self.processed_data_dir = os.path.join(self.data_dir, 'processed-sms')
        self.tableau_inputs = os.path.join(self.data_dir, 'tableau-inputs')
        self.box_file = os.path.join(data_dir, 'Distribution_Boxes@{}.xlsx'.format(box_dist_ver))
        self.observed_sms_raw_file = os.path.join(data_dir, 'processed-sms', 'sms_observed.csv')
        self.observed_sms_valid_file = os.path.join(data_dir, 'processed-sms', 'sms_observed_valid.csv')
        self.sms_rect_file = os.path.join(data_dir, 'processed-sms', 'sms_rect_hr.csv')
        self.stata_powerout_file = os.path.join(data_dir, 'powerout.csv')
        self.daily_outage_file_actual = os.path.join(data_dir, 'imputation-verification',
                                                     'daily_outage_duration_actual.csv')
        self.daily_outage_file_imputed = os.path.join(data_dir, 'imputation-verification',
                                                      'daily_outage_duration_imputed.csv')
        self.processing_msgs_file = os.path.join(data_dir, 'processed-sms', 'data_processing_msgs.txt')
        self.imputed_data_file = os.path.join(data_dir, 'processed-sms', 'sms_rect_hr_imputed.csv')
        self.box_metadata = None
        self.psu_coords_file = os.path.join(data_dir, 'PSU_Coordinates.csv')
        if verification_mode:
            self.xml_file = os.path.join(self.xml_dir, 'sms-verification.xml')
        else:
            self.xml_file = os.path.join(self.xml_dir, 'sms.xml')

    def process_data(self):
        # ===============================================
        #   BACKS UP
        # ===============================================
        if self.process_type == 'backup':
            self.backup_data()

        # ===============================================
        #   SAVE RAW SMS EVENTS
        # ===============================================
        if self.process_type == 'save-all-events':
            self.save_all_observed_sms_events()

        # ===============================================
        #   SAVE VALID EVENTS AND RECTANGULAR FILE
        # ===============================================
        if self.process_type == 'valid-and-rectangular-events':
            self.save_valid_sms_events_and_rectangular_dataset()
            
        # ===============================================
        #   OUTAGE SUMMARY
        # ===============================================
        if self.process_type == 'outage-summary':
            self.save_daily_outage_duration()

    def backup_data(self):
        """
        Backs up data
        :return:
        """
        now = datetime.now()
        ut.convert_xml_to_csv(self, ts=now)

    def save_observed_events_dataset(self, valid_events_only=False):
        """
        Saves hourly based dataset
        :return:
        """
        if valid_events_only:
            print('Saving only valid observed events....'+'\n')
        else:
            print('Saving all observed events, including invalid ones....' + '\n')

        box_objects = self.create_box_obj_from_events(generate_hr_events=False, after_event_threshold=13,
                                                      is_invalid_threshold=72)
        df_actual = None
        for i, obj in enumerate(box_objects.values()):
            if valid_events_only:
                obj.drop_events()

            if not valid_events_only:
                obj.valid_duplicate_events = 'NA'  # for the raw data file

            if i % 10 == 0:
                print('{} boxes processed....'.format(i))

            if i == 0:
                df_actual = obj.dataframe_from_actual_events()

            if i > 0:
                df_actual_temp = obj.dataframe_from_actual_events()
                df_actual = df_actual.append(df_actual_temp)

            i += 1
        if valid_events_only:
            df_actual.to_csv(self.observed_sms_valid_file, index=False)
        else:
            df_actual.to_csv(self.observed_sms_raw_file, index=False)

    def save_rectangularised_dataset(self):
        """
        Saves hourly based dataset
        :return:
        """
        print('Saving rectangular events...')
        box_objects = self.create_box_obj_from_events(generate_hr_events=True, after_event_threshold=13,
                                                      is_invalid_threshold=72)

        df_hr = None
        for i, obj in enumerate(box_objects.values()):

            if i % 10 == 0:
                print('{} boxes processed....'.format(i))

            if i == 0:
                df_hr = obj.dataframe_from_hourly_events()

            if i > 0:
                df_hr = df_hr.append(obj.dataframe_from_hourly_events())

            i += 1

        df_hr.to_csv(self.sms_rect_file, index=False)

    def save_all_observed_sms_events(self):
        """
        Saves all events including invalid ones as follows:
        - sms_observed.csv
        :return:
        """
        # save raw observed-sms-including invalid events
        starttime = datetime.now()
        self.save_observed_events_dataset(valid_events_only=False)
        endtime = datetime.now()
        print('Processing took {} seconds '.format((endtime - starttime).total_seconds()))

    def save_valid_sms_events_and_rectangular_dataset(self):
        """
        Saves the following datasets:
         - sms_observed_valid.csv
         - sms_rect_hr.csv
         -daily_outage_durations.csv
        :return:
        """
        # save valid events, rectangular events and outage summary
        starttime = datetime.now()
        self.save_observed_events_dataset(valid_events_only=True)
        self.save_rectangularised_dataset()
        endtime = datetime.now()
        print('Processing took {} seconds '.format((endtime - starttime).total_seconds()))

    def save_daily_outage_duration(self):
        print('Saving outage summaries....')
        df_actual = None
        box_objects = self.create_box_obj_from_events(generate_hr_events=True)
        for i, obj in enumerate(box_objects.values()):
            if obj.hourly_events.__len__() == 0:
                continue

            if i % 10 == 0:
                print('{} boxes processed....'.format(i))

            if i == 0:
                df_actual = obj.generate_daily_power_outage_duration()

            if i > 0:
                df_actual_temp = obj.generate_daily_power_outage_duration()
                df_actual = df_actual.append(df_actual_temp)

            i += 1

        df_actual.to_csv(self.daily_outage_file, index=False)

    def create_tableau_files(which=None, config_object=None, source_file='sms_rect_hr_imputed.csv'):
        """
        Using either sms_rect_hr_imputed or other file, we save file into those required for tableau
        :param which:
        :param config_object:
        :return:
        """

        if which == 'powerout':
            input_file = os.path.join(config_object.get_processed_data_dir(), 'sms_rect_hr_imputed.csv')
            df = pd.read_csv(input_file)
            # sort events
            df = df.sort_values(by=['box_id', 'str_datetime_sent_hr'])

            # ------------------ADD AND RENAME COLUMNS ----------------
            # this id done to match columns in the original powerout.csv
            df['dhms'] = df.str_datetime_sent_hr
            df['date_powerfailure'] = df.date_sent
            df['date_powerfailure_hour'] = df.str_datetime_sent_hr

            # set when power comes back
            df.loc[df['event_type'] == 'pback', 'date_powerback'] = df.date_sent
            df.loc[df['event_type'] == 'pback', 'date_powerback_hour'] = df.str_datetime_sent_hr

            df['POWERout'] = 0
            df.loc[df['power_state'] == 0, 'POWERout'] = 1
            df.loc[df['power_state'] == 99, 'POWERout'] = 99

            df.rename(columns={'box_id': 'BoxID', 'date_collection_started': 'DateCollectionStart'})

            return None

    def box_metadata_from_dist_and_psu_coordinates(self, true_num_boxes=NUM_WORKING_BOXES):
        """
        Use pandas to read in 'Distribution_Boxes@version.xlxs and psu_coordinates which contain boxes details and
        output it as  dict.
        :return: a nested dict like so- {'box_id: {box metadata}}
        """

        df_coords = pd.read_csv(self.psu_coords_file)
        df_box = pd.read_excel(self.box_file, sheetname='Лист1')
        useful_cols = ['ClusterId', 'District/City', 'Date Collection Start', 'BoxID',
                       'Phone number']
        df_box = df_box[useful_cols]

        bx = pd.merge(left=df_coords, right=df_box, left_on='PSU_ID', right_on='ClusterId')
        cols_to_keep = ['URB_RUR', 'REGION', 'LONG', 'LAT', 'ClusterId', 'District/City',
                        'Date Collection Start', 'BoxID', 'Phone number']
        bx = bx[cols_to_keep]
        bx.rename(columns={'ClusterId': 'psu', 'LONG': 'lon', 'LAT': 'lat', 'BoxID': 'box_id', 'REGION': 'region',
                           'Phone number': 'phone_number', 'Date Collection Start': 'DateCollectionStart',
                           'URB_RUR': 'urb_rur', 'District/City': 'district'}, inplace=True)

        # Ensure that they are 298 unique boxes
        num_boxes = len(list(bx['box_id'].unique()))
        try:
            assert (num_boxes == true_num_boxes)
        except AssertionError as e:
            print('*** Please check the number of boxes **')

        # Fix problematic dates-strictly based on Will's do file

        # Convert Dataframe to dictionary-first make box_id
        bx.set_index('box_id', inplace=True)

        # Now convert to dict and return it
        self.box_metadata = bx.to_dict(orient='index')

    def create_box_obj_from_events(self, after_event_threshold=13, is_invalid_threshold=72, generate_hr_events=False):
        """
        Creates box objects and events for the box. Also, computes all necessary properties for the box.
        The insertion procedure is based on previous event.
        :param xml_file: raw sms.xml file
        :param box_metadata: Box details to get info such as lat/lon
        :param after_event_threshold: For use in inserting events
        :param is_invalid_threshold:
        :param debug_mode: Debug or not
        :param processing_msgs_file: Important processing messages
        :param generate_hr_events: whether to create rectangular dataset
        :param box_details_file: Document details of box file used in processing
        :return:
        """

        # create box-metadata if its not there
        if not self.box_metadata:
            self.box_metadata_from_dist_and_psu_coordinates()

        # convert sms.xml into dict objects for (hopefully) faster processing
        box_events = ut.event_dict_from_xml(self.xml_file)
        box_objects = {}  # holds box objects which we will return

        # keep invalid messages here
        box_with_few_events = []  # lst to keep boxes with no events

        # when debug mode is True, this counter is used to stop at 10 or 50 boxes
        box_count = 0

        # the date is put there to go around
        dummy_date = ut.convert_to_aware_datetime_object(date_str='9/1/2016 12:00:00')
        self.box_metadata.update({-99: {'DateCollectionStart': dummy_date, 'psu': 999, 'URB_RUR': 999, 'lon': 99, 'lat': 999,
                                   'region': 999, 'district': 999}})
        event_cnts = 0
        for bx_id, raw_bx_events in box_events.items():
            # if bx_id not in [1055, 1039, 1249, 20, 525]:
            #     continue

            # initialise a box object
            box_meta = self.box_metadata.get(bx_id, None)

            if box_meta:
                box_obj = ut.Box(date_collection_started=box_meta.get('DateCollectionStart'),
                              box_id=bx_id, psu=box_meta.get('psu'), lon=box_meta.get('lon'),
                              urban_rural=box_meta.get('URB_RUR'),
                              lat=box_meta.get('lat'), district=box_meta.get('district'), region=box_meta.get('region'))
            else:
                dummy_date = ut.convert_to_aware_datetime_object(date_str='9/1/2016 12:00:00')
                box_obj = ut.Box(date_collection_started=dummy_date, box_id=bx_id, psu=999, lon=999, urban_rural=999,
                              lat=999, district=999, region='999')
                box_obj.valid_box_id = False

            # generate event objects from raw sms messages
            box_obj = ut.generate_event_objects(box_raw_events=raw_bx_events, box_obj=box_obj)
            event_cnts += len(box_obj.actual_events)

            # check number of events-skip boxes with less than 5 messages
            if len(box_obj.actual_events) < 1:
                box_with_few_events.append(bx_id)
                print('SKIPPING BOX {} BECAUSE IT HAS ONLY {} EVENTS..'.format(bx_id, len(box_obj.actual_events)))
                continue

            # Generate hourly events-could take long
            if generate_hr_events:
                box_obj.generate_hourly_events_based_on_power_state(after_event_threshold=after_event_threshold,
                                                                    invalid_threshold=is_invalid_threshold)
                box_obj.check_number_of_hr_events()

            # Add event object to Box object
            box_objects[bx_id] = box_obj

            box_count += 1
            # In debug mode only create 10 objects
            if self.debug_mode:
                if box_count == 10:
                    break

        # write log-file for skipped events
        # log_processing_message(msgs_with_no_id=no_id_msgs, outfile=processing_msgs_file, test_events=test_events,
        #                        start_date_events=events_before_start, out_csv_file=dropped_ev_csv,
        #                        box_file=box_details_file, boxes_with_no_events=box_with_few_events)

        print('Boxes with no events: {}'.format(box_with_few_events))
        print('Number of boxes done: {}'.format(box_count))
        print('Events: {}'.format(event_cnts))
        return box_objects


class Imputation:
    """
    Handles mechanics of the imputation
    """
    def __init__(self, method=None, imputation_type=None, imputation_model_params=None, model_name=None,
                 keep_invalid_events=None, data_processor: DataProcessor=None, tolerance=None, verify=False):
        self.imputation_approach = method
        self.imputation_type = imputation_type
        self.model_params = imputation_model_params
        self.model_name = model_name
        self.keep_invalid = keep_invalid_events
        self.dp_object = data_processor
        self.tolerance = tolerance
        self.verify = verify

    def impute(self):
        if self.imputation_type == 'on-demand':
            self.impute_on_demand()
        self.daily_averages_for_imputed_dataset()
        if self.verify:
            self.check_imputed_file()

    def impute_on_demand(self):
        """
        Simplest case, we impute on demand, overwrite the existing files
        :return:
        """
        if os.path.exists(self.dp_object.sms_rect_file):
            self.impute_with_out_of_the_box_model(model_name=self.model_name, predictor_params=self.model_params,
                                                  keep_invalid=True)
        else:
            print('Rectangular file not available')

    def impute_with_universal_model(self, how='nn'):
        """
        Fills in missing values using either nearest neighbor(nn) or an out of box model such as random forest
        and saves result as 'sms_v3.csv'.
        Here, we use same parameters and model for all boxes.
        :param params:
        :return:
        """

        if how == 'nn':
            # --------impute with nearest neighbor--------------------
            nn_params = {'neighbors': 1, 'time-window': 7, 'direction': 'both', 'how': 'frequent'}
            df_imputed = impute_with_nearest_neighbor(file_sms_rect_hr=self.dp_object.sms_rect_file, predictor_params=nn_params,
                                                      bx_file=self.dp_object.box_file)

            df_imputed.to_csv(self.dp_object.imputed_data_file, index=False) # save to file
        elif how == 'out':
            # --------impute with out of the box model--------------------
            model = 'RF'
            prediction_features = ['box_id', 'psu', 'lon', 'lat', 'month_sent', 'wk_day_sent', 'wk_end', 'holiday']
            params = {'trees': 1, 'pred_feat': prediction_features}
            self.impute_with_out_of_the_box_model(predictor_params=params, model_name=model, keep_invalid=False)

    def impute_with_out_of_the_box_model(self, predictor_params=None, model_name='ETC', keep_invalid=True):
        """
        :param predictor_params: Parameters for predictors-to be sourced dynamically from file
        :param model_name:
        :param keep_invalid:
        :return: Saves imputed data file to csv
        """
        # -------------PICK MODEL----------------------------------------------------
        random_state = 1
        clfs = {'LR': LogisticRegression(penalty='l1', max_iter=50),
                'RF': RandomForestClassifier(n_estimators=500, n_jobs=-1,
                                             random_state=random_state),
                'GBM': GradientBoostingClassifier(n_estimators=500,
                                                  random_state=random_state),
                'ETC': ExtraTreesClassifier(n_estimators=100, n_jobs=-1)
                }

        clf = clfs[model_name]
        pred_feat = predictor_params['pred_feat']

        # -------------READ IN DATA----------------------------------------------------
        cols_to_use = set(['box_id', 'psu', 'lon', 'lat', 'datetime_sent_raw', 'str_datetime_sent', 'date_sent',
                           'str_datetime_sent_hr','hour_sent', 'event_type_str', 'power_state', 'data_source',
                           'date_collection_started'] + pred_feat)

        df = pd.read_csv(self.dp_object.sms_rect_file, usecols=list(cols_to_use))

        target_var = predictor_params['target_var']
        num_neighbors = predictor_params['neighbors']
        window_len = predictor_params['window_len']

        # in case we want to train the model on specific dates and on specifc neighbors
        if num_neighbors != -1 or window_len != -1:
            # handle selection of training data here
            print('We probably wont bother')

        train_df = df[df[target_var].isin([1, 0])]  # Keep only non-missing values for evaluation
        df_missing = df[df[target_var] == -1]

        df_missing.is_copy = False
        print('{} missing values to be computed out of {} total events.'.format(df_missing.shape[0], df.shape[0]))

        # -------------TRAIN MODEL------------------------------------------------------
        x = train_df[pred_feat].values
        y = train_df[target_var].values

        clf.fit(X=x, y=y)

        # ------------IMPUTE-------------------------------------------------------------
        x_predict = df_missing[pred_feat].values
        y_predicted = clf.predict(x_predict)

        # ----------REPLACE MISSING VALUES WITH PREDICTED VALUES IN DF-------------------
        df.loc[df[target_var] == -1, 'data_source'] = 'imputed'
        df.loc[df[target_var] == -1, target_var] = y_predicted

        # ----------SAVE UPDATED DATAFRAME-----------------------------------------------
        # keep or drop invalid rows
        # columns to save
        cols_to_save = ['box_id', 'psu', 'lon', 'lat', 'date_collection_started', 'date_sent','datetime_sent_raw',
                        'str_datetime_sent', 'str_datetime_sent_hr', 'hour_sent', 'event_type_str', 'power_state',
                        'data_source']
        df = df[cols_to_save]
        df.rename(columns={'str_datetime_sent': 'datetime_sent', 'str_datetime_sent_hr': 'datetime_sent_hr',
                           'event_type_str': 'event_type'})
        if keep_invalid:
            df.to_csv(self.dp_object.imputed_data_file, index=False)
        else:
            df = df[df[target_var] != 99]
            df.to_csv(self.dp_object.imputed_data_file, index=False)

        # Check that they are no missing values anymore
        num_missing = df[df[target_var] == -1].shape[0]
        if num_missing == 0:
            print('Successfully finished imputations')

    def check_imputed_file(self):
        """
        Simple test to ensure imputed file is okay
        :return:
        """
        # Read the sms_rect_hr files (before and after imputations)
        df_before = pd.read_csv(self.dp_object.sms_rect_file)
        df_after = pd.read_csv(self.dp_object.imputed_data_file)

        # Ensure that number of events hasn't changed between the two files
        try:
            assert (df_before.shape[0] == df_after.shape[0])
        except AssertionError:
            print('The input dataset and imputed datasets have different dims')
            print('before-imputation: {}, after-imputation: {}'.format(df_before.shape[0], df_after.shape[0]))

        # Ensure they are no missing values
        num_missing = df_after[df_after.power_state == -1].shape[0]
        try:
            assert (num_missing == 0)
        except AssertionError:
            print('There should be no  missing values')

        # Check the daily mean differences between imputed and actual
        df_diff_py, df_diff_stata = self.compare_outage_summaries_imputed_vs_actual()
        # mean_diff_py = df_diff_py['abs_diff'].mean()
        mean_diff_stata = df_diff_stata['abs_diff'].mean()

        try:
            assert (mean_diff_stata <= self.tolerance)
            print(' The mean difference is {} hours. Its within acceptable range'.format(mean_diff_stata))
        except AssertionError:
            print(' The mean difference is {} hours. Its way too large'.format(mean_diff_stata))

    def daily_averages_for_imputed_dataset(self):
        df = pd.read_csv(self.dp_object.imputed_data_file)
        # Ensure they are no invalid files
        df = df[df.power_state != 99]

        # summarise
        df_all = None
        i = 0
        for bx in list(df.box_id.unique()):
            df_bx = df[df.box_id == bx]
            df_bx.is_copy = False

            if i == 0:
                df_all = ut.summarize_outages_by_date(df=df_bx, box_id=bx)

            if i > 0:
                df_bx_sum = ut.summarize_outages_by_date(df=df_bx, box_id=bx)
                df_all = df_all.append(df_bx_sum)

            i += 1

        df_all.to_csv(self.dp_object.daily_outage_file_imputed, index=False)

    @staticmethod
    def compare_outage_summaries(df_sum_actual=None, df_sum_imp=None, target_var=None):
        """
        
        :param df_sum_actual:
        :param df_sum_imp:
        :param target_var:
        :return:
        """
        # combine them
        suffixes = {'actual': '_actual', 'imputed': '_imputed'}
        df = pd.merge(left=df_sum_actual, right=df_sum_imp, suffixes=[suffixes['actual'], suffixes['imputed']],
                      on='day', how='inner', indicator=True)

        # add absolute difference column
        col_name_imp = 'avg_' + target_var + suffixes['imputed']
        col_name_actual = 'avg_' + target_var + suffixes['actual']

        df['abs_diff'] = df.apply(lambda x: abs(x[col_name_imp] - x[col_name_actual]), axis=1)

        # save this file
        return df

    def compare_outage_summaries_imputed_vs_actual(self, target_var='hrs_power_off'):
        """
        :return:
        """
        # Get outage daily outage summaries for each separate file and plot
        df_imp = pd.read_csv(self.dp_object.daily_outage_file_imputed)
        sum_imp = ut.calculate_daily_average_power_out(df=df_imp, var=target_var)
        self.plot_average_outage_summary_single(df=sum_imp, out_file_name='avg_outage_imputed.png')

        df_actual = pd.read_csv(self.dp_object.daily_outage_file_actual)
        sum_actual = ut.calculate_daily_average_power_out(df=df_actual, var=target_var)
        self.plot_average_outage_summary_single(df=sum_actual, out_file_name='avg_outage_actual_from_python.png')

        df_stata = ut.summarise_outage_counts_from_stata_powerout_file(powerout_file=self.dp_object.stata_powerout_file)
        out_file = os.path.join(self.dp_object.data_dir, 'imputation-verification',
                                'daily_outage_duration_actual_stata.csv')
        df_stata.to_csv(out_file, index=False)
        sum_stata = ut.calculate_daily_average_power_out(df=df_stata, var=target_var)
        self.plot_average_outage_summary_single(df=sum_stata, out_file_name='avg_outage_actual_from_stata.png')

        # compare outage duration for each day between imputed file and
        # actual durations from Python and Stata
        diffs_py = self.compare_outage_summaries(df_sum_actual=sum_actual, df_sum_imp=sum_imp, target_var=target_var)
        out_file_py_diffs = os.path.join(self.dp_object.data_dir, 'imputation-verification',
                                            'outage_duration_diffs_imputed_vs_python_actual.csv')
        diffs_py.to_csv(out_file_py_diffs, index=False)
        # plot both on a single plot
        self.plot_average_outage_summary_multiple(diffs_py, outfile='python_imputed_file_vs_python_actual.png')

        # compare python imputed file vs stata powerout file
        diffs_stata = self.compare_outage_summaries(df_sum_actual=sum_stata, df_sum_imp=sum_imp, target_var=target_var)
        out_file_stata_diffs = os.path.join(self.dp_object.data_dir,'imputation-verification',
                                'outage_duration_diffs_imputed_vs_stata_actual.csv')
        diffs_stata.to_csv(out_file_stata_diffs, index=False)
        self.plot_average_outage_summary_multiple(diffs_stata, outfile='python_imputed_file_vs_stata_actual.png')

        return diffs_py, diffs_stata
        
    def plot_average_outage_summary_multiple(self, df=None, outfile=None):
        """
        Plots differences for visual display
        :return:
        """
        plt.figure()
        ax = plt.subplot(111)
        ax.figure.set_size_inches(15, 8)
        plt.plot(df.avg_hrs_power_off_actual, label="Actual")
        plt.plot(df.avg_hrs_power_off_imputed, label="Imputed")
        leg = plt.legend(loc='upper right', ncol=1, mode="expand", shadow=True, fancybox=True)
        leg.get_frame().set_alpha(0.5)
        ax.set_title('Compare daily average summary between actual and imputed rectangular dataset')
        ax.set_xlabel('Day')
        ax.set_ylabel('Daily average outage duration in hrs')

        output_file = os.path.join(self.dp_object.outputs_dir, 'imputation-verification', outfile)
        ax.figure.savefig(output_file)

    def plot_average_outage_summary_single(self, df=None, out_file_name=None):
        """
        Plot average outage summary for a single variable
        :return:
        """
        plt.figure()
        plt.plot(df.avg_hrs_power_off)
        plt.title('Daily average outage duration')
        plt.xlabel('Day')
        plt.ylabel('Daily average outage duration in hrs')

        output_file = os.path.join(self.dp_object.outputs_dir, 'imputation-verification', out_file_name)
        plt.savefig(output_file)


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


def impute_with_gradient_boosted_trees(config_obj=None, prediction_features=None, model_name=None):
    """
    Fill out missing events using scikit-learn Extra Trees Classifier based.
    :param config_obj:
    :param prediction_features:
    :return:
    """
    file_sms_rect_hr = config_obj.get_processed_data_dir() + 'sms_rect_hr.csv'

    model_file = config_obj.get_model_dir() + model_name
    file_sms_rect_hr_imp = config_obj.get_processed_data_dir() + 'sms_rect_hr_imp.csv'

    df = pd.read_csv(file_sms_rect_hr)

    df_missing = df[df['power_state'] == -1]

    if not prediction_features:
        prediction_features = ['box_id', 'psu', 'lon', 'lat', 'hour_sent', 'month_sent', 'day_sent', 'wk_day_sent',
                           'wk_end']

    X = df_missing[prediction_features]

    try:
        clf = joblib.load(model_file)

        y_predicted = clf.predict(X)

        df.loc[df['power_state'] == -1, 'power_state'] = y_predicted

        df.to_csv(file_sms_rect_hr_imp, index=False)

    except Exception as e:
        print(e)

    print('SUCCESSFULLY IMPUTED WITH GBM')