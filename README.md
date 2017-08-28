*Disclaimer: The World Bank takes no responsibility for anything in this repo etc luctus quis tortor. 

Smart-Survey-Boxes: Listening to Tajikistan
=======

The code in this repository is based on the Listening to Tajikistan project. The purpose of the project is 
to monitor energy grid reliability for households by monitoring power failures. Hence the name **powermon**.
In October 2016, smart survey boxes were installed in 300 households. Since then, the boxes have been logging 
power failures and sending the data to a Google Drive server. This repository contains code for preprocessing the data, 
imputing missing values in the data and doing out of sample estimations.

Key components
----------------------
This project has the following components:

* Data preprocessing
* Imputation
* Out of sample prediction
* Satellite night lights integration


Folder Structure for the project
--------
Here is an overview of the project files and their descriptions. This is a sample structure.

    powermon
    |- README          # This readme
    |- outputs/        # Outputs from analysis and model evaluation
    |- power-maps/     # interactive mapping app for out of sample predicted variables
    |- power-viz/      # javascript based visualisation of different variables
    |- pypower/        # python package for data preprocessing and model evaluation
   
How to use
----------
Download or clone to this repository to your computer.
Navigate to the pypower directory. Make changes to the configuration object to ensure 
you are pointing to the correct folders for input. These instructions assume you have Python 3.6
or above, if you dont have it, please install and follow the rest of the instructions.

````
pip install -r requirements.txt

````
Further documentation
--------------------------------
Please refer to our wiki page for detailed instructions.
