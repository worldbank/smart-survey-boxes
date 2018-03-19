"""
Imputes missing events
"""
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
from data_processing import data_processing_engine as dp


def main():
    # grab environment variables (data folders etc) from dp
    env_vars = dp.ENV_VARS

    # Initiate a data processor object
    data_processor_params = {'data_dir': env_vars['data_folder'], 'process_type': 'impute', 'verification_mode': False,
                             'debug_mode': False, 'xml_dir': env_vars['xml_folder'],
                             'outputs_dir': env_vars['outputs_folder'],
                             'box_dist_ver': env_vars['box_dist_ver']}

    dp_obj = dp.DataProcessor(**data_processor_params)

    params = {'pred_feat': ['lon', 'lat', 'hour_sent', 'month_sent', 'day_sent', 'wk_day_sent', 'wk_end'],
              'target_var': 'power_state', 'window_len': -1, 'neighbors': -1}
    imputor = dp.Imputation(imputation_model_params=params, imputation_type='on-demand', model_name='ETC',
                            keep_invalid_events=True, data_processor=dp_obj, tolerance=2,verify=False)
    imputor.impute()


if __name__ == "__main__":
    main()
