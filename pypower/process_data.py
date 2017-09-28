"""
Runs once a day: backs up xms as csv, preprocess data and saves as actual events with variables,
imputes missing values.
"""
from datetime import datetime

from pypower import preprocessing as prep


if __name__ == "__main__":
    conf = prep.Configurations(platform='mac', imputation_approach='etc', debug_mode=False)
    debug = conf.debug_mode

    now = datetime.now()

    # ======== DATA PROCESSING =========================
    # prep.preprocesss_raw_sms(conf, debugging=debug)

    # ======== IMPUTE MISSING DATA =====================
    prep.impute_with_gradient_boosted_trees(conf, model_name='gbm.pkl')
