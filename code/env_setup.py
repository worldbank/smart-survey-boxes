"""
Sets up the environment first time
"""
import os
import pickle
import collections
import shutil


ENV = collections.namedtuple('ENV', 'project_dir xml_source_dir box_dist_ver')


def get_env_variables():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    env_object_supposed_path = os.path.join(curr_dir, 'env_vars.pkl')

    if os.path.exists(env_object_supposed_path):
        with open(env_object_supposed_path, 'rb') as input_obj:
            env_vars = pickle.load(input_obj)

            # Ensure that data folder has PSU_coordinates.csv and Distribution_Boxes@14.xlsx
            check_for_required_files_in_data(proj_folder=env_vars.project_dir, file_name='PSU_coordinates.csv')
            check_for_required_files_in_data(proj_folder=env_vars.project_dir, file_name='Distribution_Boxes@14.xlsx')

    else:
        env_vars = get_directories_from_user()
        with open(env_object_supposed_path, 'wb') as output:  # Overwrites any existing file.
                pickle.dump(env_vars, output, pickle.HIGHEST_PROTOCOL)

    return env_vars


def create_dir_safely(dir_name):
    """

    :param dir_name:
    :return:
    """
    try:
        os.makedirs(dir_name)
    except OSError:
        if not os.path.isdir(dir_name):
            print('Please make sure the path for project folder is specified correctly...')


def check_for_required_files_in_data(proj_folder=None, file_name=None):
    # Check if data folder has required input files
    repo_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    user_data_dir = os.path.abspath(proj_folder + '/data/')
    req_file_path = os.path.join(user_data_dir, file_name)
    if not os.path.exists(req_file_path):
        print('Copying {} from {} to {}'.format(file_name, repo_data_dir, user_data_dir))
        shutil.copy(os.path.join(repo_data_dir, file_name), user_data_dir)
        if file_name == 'Distribution_Boxes@14.xlsx':
            print('See WARNING below about Distribution_Boxes@14.xlsx !!...')
            print('Please note that this file may not be the latest, check with William for the latest file.')


def get_directories_from_user():
    """
    If those variables are None, ask user to prpvide them
    :return:
    """
    proj_folder = input("Provide full path of project folder: ")
    xml_folder = input("Provide full path for SMSBuckStore where sms.xml is located...: ")

    # Check that the path for xml is correct
    if not os.path.exists(xml_folder):
        print('Provide correct path..... please check the following:')
        print('1. Path is not enclosed in quotes')
        print('2. You have a directory where you are keeping the xml files')

    # Create required project folders
    folders_to_create = [proj_folder, proj_folder + '/data/processed-sms', proj_folder + '/data/raw-sms-backup',
                         proj_folder + '/data/imputation-verification', proj_folder + 'outputs']

    for f in folders_to_create:
        create_dir_safely(f)

    # Ensure that data folder has PSU_coordinates.csv and Distribution_Boxes@14.xlsx
    check_for_required_files_in_data(proj_folder=proj_folder, file_name='PSU_coordinates.csv')
    check_for_required_files_in_data(proj_folder=proj_folder, file_name='Distribution_Boxes@14.xlsx')

    out_vars = ENV(project_dir=proj_folder, xml_source_dir=xml_folder, box_dist_ver=14)

    return out_vars


if __name__ == '__main__':
    evars = get_env_variables()
    print(evars)
