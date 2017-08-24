"""
Runs once a day to convert xml file to csv.
"""
from datetime import datetime

from pypower import data_utils as ut
from pypower import preprocessing as prep


if __name__ == "__main__":
    conf = prep.Configurations(platform='bank_windows', imputation_approach='etc', debug_mode=False)
    debug = conf.debug_mode

    now = datetime.now()

    # ======== DATA BACK-UP   ==========================
    ut.convert_xml_to_csv(config=conf, ts=now)