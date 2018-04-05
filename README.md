*Disclaimer: The World Bank takes no responsibility for anything in this repo etc luctus quis tortor. 

Smart-Survey-Boxes: Listening to Tajikistan
============================================

The code in this repository is based on the Listening to Tajikistan project. The purpose of the project is 
to monitor energy grid reliability for households by monitoring power failures. 
In October 2016, smart survey boxes were installed in 300 households. Since then, the boxes have been logging 
power failures and sending the data to a Google Drive server. This repository contains code for processing the data, 
imputing missing values in the data and doing out of sample estimations.

Key Components
============================================
This system provides an *end-to-end* platform for processing the sms data coming from the 
devices. The system has three main components as follows:
- **Daily sms backup:** Converts raw sms data from xml to csv format
- **Sms data processsing:** All the core processing is done at this stage.
- **Impute missing values:** For hours where an event was missing, those evemts are imputed at this stage 

Folder Structure for the Project
============================================
Below is an overview of the project files and their descriptions. This is a sample structure.

    smart-survey-boxes
    |- README                   # This readme
    |- data/        	        # Not shown here but thid folder contains all the data
    |- code/        	        # Contains all the python and STATA code
        |- analysis_scripts/    # miscellenous analysis scripts
        |- data_processing/     # data processing scripts
        |- map-app/             # interactive mapping app for out of sample predictions
    |- docs/        	        # Project documents and other resources
    |- venv/                    # Required Python environment to run the code
    |- outputs/                 # Outputs from analysis
    
   
Installation and Setup
============================================

**Requirements**

This repository is based on Python 3. If you don't have Python, the fastest and easiest  way to install Python and the required packages is through [Anaconda](https://www.anaconda.com/what-is-anaconda/). For those working on WBG windows computers, please install Anaconda through e-services, once installed take note of the path for the Python interpreter. Anaconda will come with all the libraries which have been used in this project.

**Download the Repository**

Download or clone this repository to your computer. This repository is still private, so you need to sign up for World Bank GitHub account in order to be able to access this repo on GitHub.com.   Once downloaded, unzip the file if necessary.

**Environment Setup**

The package requires three main environment settings as described here. First, navigate to the *code* directory. While there, open *env_setup.py* in any text editor and edit the variables described below:
- **source of xml data:** Currently, the sms data is being held on Google Drive. If you have access to this shared folder, take note of its full path. In the Python file *env_setup.py*, at the top of the file, edit value for the variable XML_DIR.
- **Project folder:** The desired folder structure should the same the one  for this repository. Therefore, its recommended that you keep this repo folder as your project folder. However, you can choose another existing/new project folder. This is the folder where all the data outputs will be saved.If you don't provide a separate project folder, the repository folder will be used as the project folder. In the Python file *env_setup.py*, at the top of the file, edit value for the variable PROJECT_DIR by providing full path to your desired project folder.
- **Box Distribution version:** The location of each box and other details are saved in a file with name like this one *Distribution_Boxes@14.xlsx* where the 2 digits at the end indicates the version of the file. You need to provide that version here. For now, the version is 14.In the Python file *env_setup.py*, at the top of the file, edit value for the variable BOX_DIST_VER.

Running the Code
============================================

#### Which scripts to run?
They are four python scripts which can/should be run in this older:
   
1. **code/data_processing/daily_data_backup.py:**  this is meant to convert raw sms.xml to csv. The output from this script is a csv file which will be saved in *./data/raw-sms-backup/* with name of that day's date.
2. **code/data_processing/process_raw_sms_events.py:** this script will output three files into the folder *./data/processed-sms/*. The files are *sms_observed_valid.csv*, *sms_observed.csv* and *sms_rect_hr.csv* For details of contents of these file, please refer to the documentation on data processing.
3. **code/data_processing/process_tableau_files:** outputs file to be used in tableau dashboard. The file is in *./data/processed-sms/* with name *powerout_duration.csv*
4. **code/data_processing/impute.py:** Imputes missing values and outputs this file *./data/processed-sms/sms_rect_hr_imputed*
 
#### How to run the scripts
The scripts can be run *on demand* on the command line. For windows 10, the process is as follows:
1. Invoke powershell
2. Navigate to smart-survey-boxes/code/data_processing/
3. If Anaconda was installed properly, then you can invoke Python interpreter by just typing *python*, therefore you can run any of the scripts as below:
````
python impute.py

````  
If the above doesnt work, then you may need to add use the full path to Python interpreter. In my case, its like below:
````
C:\ProgramData\Anaconda3\python.exe impute.py

````  
You can automate running of scripts in windows by using Windows scheduler. 
In this case, make sure you always use full path for the Python interpreter. 
Also, avoid spaces in file/folder names. For instance, *Google Drive* doesnt seem to work with   Windows Scheduler, instead use *Google-Drive*

Further Documentation
--------------------------------
Please refer to our wiki page for detailed instructions.
