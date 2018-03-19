"""
Saves the observed, observed_valid and rectangular dataset
"""
import sys
# A quick and dirty solution to deal with running the script in command line and using task scheduler
user_id = 'WB255520'
path = None
if user_id == 'WB454594':
    path = r'C:\Users\{}\Google-Drive\smart-survey-boxes\code'.format(user_id)
elif user_id == 'WB255520':
    path = r'C:\Users\{}\Google-Drive\smart-survey-boxes\code'.format(user_id)

from data_processing import data_processing_engine as dp


def main(save_raw_events=False):
    # grab environment variables (data folders etc) from dp
    env_vars = dp.ENV_VARS

    # Initiate a data processor object
    process = 'valid-and-rectangular-events'  # for saving valid events only
    if save_raw_events:
        process = 'save-all-events'  # for saving raw events

    # ==========================================================
    # SAVE ALL EVENTS INCLUDING INVALID ONES
    # ==========================================================
    data_processor_params = {'data_dir': env_vars['data_folder'], 'process_type': process, 'verification_mode': False,
                             'debug_mode': False, 'xml_dir': env_vars['xml_folder'],
                             'outputs_dir': env_vars['outputs_folder'],
                             'box_dist_ver': env_vars['box_dist_ver']}

    data_processor = dp.DataProcessor(**data_processor_params)
    data_processor.process_data()


if __name__ == "__main__":
    # save raw events
    main(save_raw_events=True)

    # save rectangular and valid events
    main(save_raw_events=False)





