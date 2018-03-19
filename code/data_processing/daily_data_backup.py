"""
Runs once a day to convert xml file to csv.
"""
import sys

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

    # initiate a data processor object
    data_processor_params = {'data_dir': env_vars['data_folder'], 'process_type': 'backup', 'verification_mode': False,
                             'debug_mode': False, 'xml_dir': env_vars['xml_folder'],
                             'outputs_dir': env_vars['outputs_folder'],
                             'box_dist_ver': env_vars['box_dist_ver']}

    data_processor = dp.DataProcessor(**data_processor_params)

    # backup data
    data_processor.process_data()


if __name__ == "__main__":
    main()
