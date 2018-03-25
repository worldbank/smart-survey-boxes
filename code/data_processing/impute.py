"""
Imputes missing events
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


def main():
    # grab environment variables (data folders etc) from dp
    env_vars = su.get_env_variables()
    data_folder = os.path.join(os.path.abspath(env_vars.project_dir), 'data')
    xml_folder = os.path.abspath(env_vars.xml_source_dir)
    outputs = os.path.join(os.path.abspath(env_vars.project_dir), 'outputs')

    # Initiate a data processor object
    data_processor_params = {'data_dir': data_folder, 'process_type': 'impute', 'verification_mode': False,
                             'debug_mode': False, 'xml_dir': xml_folder,'outputs_dir': outputs,
                             'box_dist_ver': env_vars.box_dist_ver}

    dp_obj = dp.DataProcessor(**data_processor_params)

    params = {'pred_feat': ['lon', 'lat', 'hour_sent', 'month_sent', 'day_sent', 'wk_day_sent', 'wk_end'],
              'target_var': 'power_state', 'window_len': -1, 'neighbors': -1}
    imputor = dp.Imputation(imputation_model_params=params, imputation_type='on-demand', model_name='ETC',
                            keep_invalid_events=True, data_processor=dp_obj, tolerance=2,verify=False)
    imputor.impute()


if __name__ == "__main__":
    main()
