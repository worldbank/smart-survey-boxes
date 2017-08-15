"""
This script will simply save raw sms data as csv every day
"""
import os
import datetime, time
from pypower import utils as ut
from pypower import preprocessing as prep


if __name__ == "__main__":
    conf = prep.config(platform='mac')
    file_name = 'sms_' + time.strftime('%m-%d-%Y') + '.csv'

    dir = "C:/Users/wb344850/WBG/William Hutchins Seitz - 01.data/raw_sms/"
    out_csv = dir + file_name
    xml_file = "C:/Users/wb344850/Google Drive/SMSBuckupRestore/sms.xml"

    while True:
        prep.xml_to_csv(os.path.normpath(xml_file), os.path.normpath(out_csv))
        time.sleep(300)  # runs once in 24 hours
