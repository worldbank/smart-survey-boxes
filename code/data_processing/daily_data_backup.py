"""
Runs once a day to convert xml file to csv.
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
        raise ImportError('Package directory not correctly added')


def main():
    # grab environment variables (data folders etc) from dp
    env_vars = su.get_env_variables()
    data_folder = os.path.join(os.path.abspath(env_vars.project_dir), 'data')
    xml_folder = os.path.abspath(env_vars.xml_source_dir)
    outputs = os.path.join(os.path.abspath(env_vars.project_dir), 'outputs')

    # initiate a data processor object
    data_processor_params = {'data_dir': data_folder, 'process_type': 'backup', 'verification_mode': False,
                             'debug_mode': False, 'xml_dir': xml_folder,'outputs_dir': outputs,
                             'box_dist_ver': env_vars.box_dist_ver}

    data_processor = dp.DataProcessor(**data_processor_params)

    # backup data
    data_processor.process_data()


if __name__ == "__main__":
    main()

