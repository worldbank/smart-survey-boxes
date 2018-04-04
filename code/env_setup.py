"""
Sets up the environment first time
"""
import os
import sys
import collections
import shutil


ENV = collections.namedtuple('ENV', 'project_dir xml_source_dir box_dist_ver')

# ==================================================
# EDIT THE FOLLOWING LINES TO CHANGE FOLDER LOCATION
# ==================================================

# if you have a separate project folder, please replace None with 'path/to/your/project_folder'
# for windows users add r at the beginning of path string like this: r'path/to/your/project_folder'
PROJECT_DIR = None

# this cant be left blank, please put path to xml folder
# for windows users add r' at the beginning of path string like this: r'path/xml'
XML_DIR = None

# this is the version for box details file. currently using 14
BOX_DIST_VER = 14


def create_dir(dir_name):
    """

    :param dir_name:
    :return:
    """
    try:
        os.makedirs(dir_name)
    except OSError:
        if not os.path.isdir(dir_name):
            print('Please make sure the path for project folder is specified correctly...')


def create_project_subfolders(project_folder=None):
    # Create required project subfolders
    folders_to_create = [project_folder, project_folder + '/data/processed-sms',
                         project_folder + '/data/raw-sms-backup',
                         project_folder + '/data/imputation-verification', project_folder + '/outputs']

    for f in folders_to_create:
        create_dir(f)


def check_for_required_folders_and_files(proj_folder=None, file_name=None):
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


def get_env_variables(project_folder=PROJECT_DIR, xml_folder=XML_DIR, box_ver=BOX_DIST_VER):
    """
    Get and set environment variables
    :param project_folder:
    :param xml_folder:
    :param box_ver:
    :return:
    """
    prj_dir = None

    # Check that XML directory is valid and contains sms.xml
    xml_file = os.path.join(xml_folder, 'sms.xml')
    if not os.path.exists(xml_file):
        print("Please provide valid path for xml at the TOP OF THIS FILE")
        sys.exit()

    # if project_folder is empty, we use current default directory and notify user
    if not project_folder:
        print('Project folder being set to current directory, please see README.md for details')
        prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        create_project_subfolders(project_folder=prj_dir)
       
        # Check for required input files
        check_for_required_folders_and_files(proj_folder=prj_dir, file_name='PSU_coordinates.csv')
        check_for_required_folders_and_files(proj_folder=prj_dir, file_name='Distribution_Boxes@14.xlsx')

    if project_folder:
        create_project_subfolders(project_folder=project_folder)
        # Ensure that data folder has PSU_coordinates.csv and Distribution_Boxes@14.xlsx
        check_for_required_folders_and_files(proj_folder=project_folder, file_name='PSU_coordinates.csv')
        check_for_required_folders_and_files(proj_folder=project_folder, file_name='Distribution_Boxes@14.xlsx')

    env_var = ENV(project_dir=prj_dir, xml_source_dir=xml_folder, box_dist_ver=box_ver)

    return env_var
