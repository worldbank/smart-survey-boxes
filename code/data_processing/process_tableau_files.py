import pandas as pd
import os
import sys

# A quick and dirty solution to deal with running the script in command line and using task scheduler
# A quick and dirty solution to deal with running the script in command line and using task scheduler
user_id = 'WB255520'
path = None
if user_id == 'WB454594':
    path = r'C:\Users\{}\Google-Drive\smart-survey-boxes\code'.format(user_id)
elif user_id == 'WB255520':
    path = r'C:\Users\{}\Google-Drive\smart-survey-boxes\code'.format(user_id)

sys.path.append(path)
from data_processing import data_processing_utils as ut


def generate_powerout_duration_file(observed_valid=None, out_file_valid=None, out_file_invalid=None):
    """
    Given sms_observed_valid.csv, generate
    :param sms_valid_events_file:
    :return:
    """

    df = pd.read_csv(observed_valid, parse_dates=['str_datetime_sent', 'str_datetime_sent_hr'])

    cols_to_keep = ['box_id', 'date_collection_started', 'date_sent', 'str_datetime_sent', 'str_datetime_sent_hr',
                    'event_type_str', 'power_state', 'data_source']

    df = df[cols_to_keep]
    df_out_valid = pd.DataFrame(columns=df.columns)
    df_out_invalid = pd.DataFrame(columns=df.columns)

    box_count = 0
    for box in list(df.box_id.unique()):
        df_bx = df[df.box_id == box]
        try:
            if 'str_datetime_sent' not in df_bx.columns:
                print('problem')

            df_box_valid = ut.powerout_duration(df=df_bx, exclude_invalid=True, time_col='str_datetime_sent')
            df_out_valid = df_out_valid.append(df_box_valid)

            df_box_invalid = ut.powerout_duration(df=df_bx, exclude_invalid=False, time_col='str_datetime_sent')
            df_out_invalid = df_out_invalid.append(df_box_invalid)
            box_count += 1
        except Exception as e:
            print('Box-ID {}, number of events {}'.format(box, df_bx.shape[0]))
            pass

    # save the dataframes to disk
    df_out_valid.to_csv(out_file_valid, index=False)
    df_out_invalid.to_csv(out_file_invalid, index=False)
    print(box_count)


def main():
    # set up environment
    data_folder = None
    if sys.platform == 'darwin':
        data_folder = os.path.abspath('/Users/dmatekenya/Google-Drive/worldbank/smart-survey-boxes/data')

    if sys.platform == 'win32':
        # windows-onedrive
        data_folder = os.path.join('C:/', 'Users', 'wb344850', 'WBG\William Hutchins Seitz - 01.Eletricity_monitoring',
                                   '01.data')

    sms_observed = os.path.join(data_folder, 'processed-sms', 'sms_observed_valid.csv')

    out_valid = os.path.join(data_folder, 'processed-sms', 'powerout_duration.csv')
    out_invalid = os.path.join(data_folder, 'processed-sms', 'powerout_duration_with_invalid.csv')

    generate_powerout_duration_file(observed_valid=sms_observed, out_file_valid=out_valid, out_file_invalid=out_invalid)


if __name__ == '__main__':
    main()
