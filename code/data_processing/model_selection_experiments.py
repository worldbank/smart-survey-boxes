import datetime
import multiprocessing

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score

import data_processing as prep

import data_utils as util
import model_selection_custom_models as mod_sel


def print_time_taken(start_time=None, end_time=None):
    minutes_taken = (end_time - start_time).total_seconds() / 60
    print()
    print('DONE WITH THE EXPERIMENT, TOOK {:.2f} MINUTES'.format(minutes_taken))


def experiment_time_periods():
    pass


def experiment_quarter(env_obj=None, target='power_state'):
    print('==========================================')
    print('    RUNNING EXPERIMENT FOR TIME PERIODS, K = 3')
    print('==========================================')

    months = [[10, 11, 12], [1, 2, 3, 4], [5, 6, 7, 8]]

    for qt in months:
        print()
        print('--------- Quarter set to {} ---------'.format(qt))
        output_file = 'results_cv10_all_events_' + str(qt) + '.csv'
        mod_sel.time_batch_evaluation_out_of_the_box_models(config_obj=env_obj, pooled=False,
                                                            exclude_inserted_events=False,
                                                            target_var=target, output_filename=output_file,
                                                            months=qt, k=3)


def experiment_months(env_obj=None, target='power_state'):
    print('==========================================')
    print('    RUNNING EXPERIMENT FOR TIME PERIODS, K = 3')
    print('==========================================')

    all_months = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]
    months = [[i] for i in all_months]

    for qt in months:
        print()
        print('--------- Month set to {} ---------'.format(qt))
        output_file = 'results_cv10_all_events_' + str(qt) + '.csv'
        mod_sel.time_batch_evaluation_out_of_the_box_models(config_obj=env_obj, pooled=False,
                                                            exclude_inserted_events=False,
                                                            target_var=target, output_filename=output_file,
                                                            months=qt, k=3)


def experiment_test_precision_power_state(env_obj=None, f=3):
    """
    Helper function for quick experimentation. This one compares results between using event_type
    or power_state
    :param config_obj:
    :return:
    """
    print('=======================================================')
    print(' EXPERIMENT: OVERALL MODEL PERFOMANCE')
    print('=======================================================')
    print()
    print(' Experiment Metadata')
    print('=======================================================')
    print('1.  Target variable==> power_state')
    print('2.  Perfomance metric==> (unweighted) average_precision')
    print('3.  Number of cross-validation folds===> {}'.format(f))

    output_file = 'cv3_precision_overall_all_data_power_state.csv'
    mod_sel.batch_evaluation_out_of_the_box_models(config_obj=env_obj, pooled=False, exclude_inserted_events=False,
                                                   target_var='power_state', output_filename=output_file, k=f,
                                                   accuracy='prec')


def experiment_event_type_classification_report(env_obj=None, f=3):
    """
    Helper function for quick experimentation. This one compares results between using event_type
    or power_state
    :param config_obj:
    :return:
    """
    print('=======================================================')
    print(' EXPERIMENT: MODEL PERFOMANCE BY EVENT TYPE ')
    print('=======================================================')
    print()
    print(' Experiment Metadata')
    print('=======================================================')
    print('1. Target variable==> event_type_num')
    print('2. Perfomance metric==> infer from confusion matrix')
    print('3. Number of cross-validation folds===> {}'.format(f))

    mod_sel.generate_classification_report(configs=env_obj, exclude_inserted_events=False,
                                           target='event_type_num', folds=f)


def experiment_test_recall_power_state(env_obj=None, f=3):
    """
    Helper function for quick experimentation. This one compares results between using event_type
    or power_state
    :param config_obj:
    :return:
    """
    print('=======================================================')
    print(' EXPERIMENT: OVERALL MODEL PERFOMANCE')
    print('=======================================================')
    print()
    print(' Experiment Metadata')
    print('=======================================================')
    print('1.  Target variable==> power_state')
    print('2.  Perfomance metric==> Recall')
    print('3.  Number of cross-validation folds===> {}'.format(f))

    output_file = 'cv3_recall_overall_all_data_power_state.csv'
    mod_sel.batch_evaluation_out_of_the_box_models(config_obj=env_obj, pooled=False, exclude_inserted_events=False,
                                                   target_var='power_state', output_filename=output_file, k=f,
                                                   accuracy='recall')


def experiment_test_accuracy_event_type(env_obj=None, f=3):
    """
    Helper function for quick experimentation. This one compares results between using event_type
    or power_state
    :param config_obj:
    :return:
    """

    print('=======================================================')
    print(' EXPERIMENT: OVERALL MODEL PERFOMANCE')
    print('=======================================================')
    print()
    print(' Experiment metadata')
    print('=======================================================')
    print('1.  Target variable==> event_type_num')
    print('2.  Perfomance metric==> accuracy')
    print('3.  Number of cross-validation folds===> {}'.format(f))

    output_file = 'cv3_acc_overall_all_data_event_type.csv'
    mod_sel.batch_evaluation_out_of_the_box_models(config_obj=env_obj, pooled=False, exclude_inserted_events=False,
                                                   target_var='event_type_num', output_filename=output_file, k=f,
                                                   accuracy='acc')


def experiment_test_accuracy_power_state(env_obj=None, f=3):
    """
    Helper function for quick experimentation. This one compares results between using event_type
    or power_state
    :param config_obj:
    :return:
    """

    print('=======================================================')
    print(' EXPERIMENT: OVERALL MODEL PERFOMANCE')
    print('=======================================================')
    print()
    print(' Experiment metadata')
    print('=======================================================')
    print('1.  Target variable==> power_state')
    print('2.  Perfomance metric==> accuracy')
    print('3.  Number of cross-validation folds===> {}'.format(f))

    output_file = 'cv3_acc_overall_all_data_power_state.csv'
    mod_sel.batch_evaluation_out_of_the_box_models(config_obj=env_obj, pooled=False, exclude_inserted_events=False,
                                                   target_var='power_state', output_filename=output_file, k=f,
                                                   accuracy='acc')


def experiment_region_separate(env_obj=None):
    """
    Check how perfomance varies by location. There are 2 main ways to do it:
    1. Split training data, evaluate models separately for each region
    2. Pool all data for  model training BUT record accuracy by time
    :param env_obj:
    :return:
    """
    print('=====================================================')
    print('    RUNNING EXPERIMENT FOR REGIONS, K = 3')
    print('=====================================================')

    box_file = env_obj.get_data_dir() + 'Boxes.csv'
    bx = util.box_loc_metadata_as_dict(box_file)
    regions = set([v.get('region') for k, v in bx.items()])

    print('-' * 50)
    print('Train and evaluate on each region')
    print('-' * 50)

    for region in regions:
        print()
        print('++++++ RESULTS FOR THIS REGION====> {} ++++++'.format(region))
        output_file = 'results_cv3_by_region_separate' + str(region) + '.csv'
        mod_sel.location_batch_evaluation_out_of_the_box_models(config_obj=env_obj, pooled=False,
                                                                exclude_inserted_events=False, target_var='power_state',
                                                                output_filename=output_file, loc_var='region', k=3,
                                                                places=[region])


def experiment_region_district_pooled(env_obj=None):
    """
    Check how perfomance varies by location. There are 2 main ways to do it:
    1. Split training data, evaluate models separately for each region
    2. Pool all data for  model training BUT record accuracy by time
    :param env_obj:
    :return:
    """
    print('=====================================================')
    print('    RUNNING EXPERIMENT FOR REGIONS & DISTRICT-POOLED TRAINING, K = 3')
    print('=====================================================')

    print('-' * 50)
    print('Train on all data BUT do location sensitive testing')
    print('-' * 50)

    mod_sel.batch_evaluation_out_of_the_box_models_pooled(config_obj=env_obj, exclude_inserted_events=False,
                                                          target_var='power_state', output_filename=None, k=3)


def try_out_tipot(conf=None, features=None, target='power_state'):
    out = conf.get_outputs_dir()
    data = pd.read_csv(conf.get_processed_data_dir() + 'sms_rect_hr.csv')

    # resample data
    data_1 = data[data['power_state'] == 1]
    data_0 = data[data['power_state'] == 0]

    data_res = data_0.append(data_1.sample(n=data_0.shape[0]))

    # drop Nan
    if not features:
        features = ['box_id', 'psu', 'lon', 'lat', 'hour_sent', 'month_sent', 'day_sent', 'wk_day_sent', 'wk_end']

    X = data_res[features].values
    y = data_res[target].values

    clf = ExtraTreesClassifier(bootstrap=False, max_features=0.9500000000000001, min_samples_leaf=1,
                               min_samples_split=10,
                               n_estimators=100)

    scores = cross_val_score(clf, X, y, cv=10)

    print(np.max(scores), np.median(scores), np.mean(scores), np.min(scores))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    #
    # tpot = TPOTClassifier(generations=100, population_size=100, verbosity=2, n_jobs=1, max_time_mins=420,
    #                       scoring='accuracy')
    #
    # tpot.fit(X_train, y_train)
    # print(tpot.score(X_test, y_test))
    # tpot.export(out + 'tpot_code.py')


def run_experiment(experiment_name=None):
    start = datetime.datetime.now()
    experiment_name
    end = datetime.datetime.now()
    print_time_taken(start_time=start, end_time=end)


if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver', force=True)
    # ----------SET UP-----------------------------------------
    config = prep.Configurations(platform='mac')
    config.debug_mode = False

    try_out_tipot(conf=config)
    # ======== EXPERIMENTS ======================
    # exp_list = {'overral_accuracy': experiment_test_accuracy_power_state(env_obj=config, f=3),
    #             'overral_precision': experiment_test_precision_power_state(env_obj=config, f=3),
    #             'overral_recall': experiment_test_recall_power_state(env_obj=config, f=3),
    #             'classification report': experiment_event_type_classification_report(env_obj=config, f=3)}

    # for name, exp in exp_list:
    #     run_experiment(exp)

    # run_experiment(experiment_event_type_classification_report(env_obj=config, f=3))

    # from_pixel_to_power_status(conf=config, target='power_state', features=['log_radiance'])

    # run_experiment(experiment_test_accuracy_event_type(env_obj=config, f=3))

    # -----------TARGET VARIABLE-----------------
    # start = datetime.datetime.now()
    # experiment_power_state_vs_event_type(env_obj=config)
    # end = datetime.datetime.now()
    # print_time_taken(start_time=start, end_time=end)

    # -----------TIME-QUARTER---------------------
    # start = datetime.datetime.now()
    # experiment_quarter(env_obj=config)
    # end = datetime.datetime.now()
    # print_time_taken(start_time=start, end_time=end)

    # -----------TIME-MONTHS---------------------
    # start = datetime.datetime.now()
    # experiment_months(env_obj=config)
    # end = datetime.datetime.now()
    # print_time_taken(start_time=start, end_time=end)

    # -----------LOCATION-REGION---------------------
    # start = datetime.datetime.now()
    # experiment_region_separate(env_obj=config)
    # end = datetime.datetime.now()
    # print_time_taken(start_time=start, end_time=end)

    # -----------POOLED TESTING---------------------
    # start = datetime.datetime.now()
    # experiment_region_district_pooled(env_obj=config)
    # end = datetime.datetime.now()
    # print_time_taken(start_time=start, end_time=end)

    # -----------AUC---------------------
    # start = datetime.datetime.now()
    # experiment_test_auc_power_state(env_obj=config, f=5)
    # end = datetime.datetime.now()
    # print_time_taken(start_time=start, end_time=end)

    # Majority classifier
    # sms2 = config.get_processed_data_dir() + 'sms_rect_hr.csv'
    # df = pd.read_csv(sms2, usecols=['power_state','event_type_str', 'event_type_num'], nrows=500000)
    # df = df[df.power_state != -1]
    # res_m = mod_sel.evaluate_majority_classifier(df, iterations=100, target_var='power_state')
    # print(res_m)
    #
    # res_r = mod_sel.evaluate_random_classifier(df, iterations=100, target_var='power_state')
    # print(res_r)
