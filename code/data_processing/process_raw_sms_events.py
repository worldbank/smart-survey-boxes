"""
Saves the observed, observed_valid and rectangular dataset
"""
import sys
import os

# in case the script is being called from commandline or scheduler, we add path to Python path
package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(package_dir)
try:
    from data_processing import data_processing_engine as dp
    from data_processing import data_processing_utils as ut
    from data_processing import prediction_models as pred
    import env_setup as su
except ImportError:
    syspath = sys.path
    if package_dir not in syspath:
            print('Package directory not correctly added')


def main(save_raw_events=False):
    # grab environment variables (data folders etc) from dp
    env_vars = su.get_env_variables()
    data_folder = os.path.join(os.path.abspath(env_vars.project_dir), 'data')
    xml_folder = os.path.abspath(env_vars.xml_source_dir)
    outputs = os.path.join(os.path.abspath(env_vars.project_dir), 'outputs')

    # Initiate a data processor object
    process = 'valid-and-rectangular-events'  # for saving valid events only
    if save_raw_events:
        process = 'save-all-events'  # for saving raw events

    # ==========================================================
    # SAVE ALL EVENTS INCLUDING INVALID ONES
    # ==========================================================
    data_processor_params = {'data_dir': data_folder, 'process_type': process, 'verification_mode': False,
                             'debug_mode': False, 'xml_dir': xml_folder, 'outputs_dir': outputs,
                             'box_dist_ver': env_vars.box_dist_ver}

    data_processor = dp.DataProcessor(**data_processor_params)
    data_processor.process_data()


if __name__ == "__main__":
    # save raw events
    main(save_raw_events=True)

    # save rectangular and valid events
    main(save_raw_events=False)





