"""
For making out of sample predictions and imputations
"""
import sys
import operator
import random
from datetime import timedelta

# A quick and dirty solution to deal with running the script in command line and using task scheduler
sys.path.append(r'C:\Users\wb344850\Google-Drive\worldbank\smart-survey-boxes\code')
import pandas as pd
from data_processing import data_processing_utils as ut


class Results:
    def __init__(self, box_id=None, model_type=None, model_name=None,non_missing_events=None,missing_events=None,
                 invalid_events=None,support=None,tot_pred_0=None, tot_pred_1=None, correct_0=None, correct_1=None,
                 actual_0=None, actual_1=None, accuracy=None, precision_0=None, precision_1=None, recall_0=None,
                 recall_1=None, window_len=None, neighbors=None,max_training_cases=None,min_training_cases=None,
                 median_training_cases=None,mean_training_cases=None, total_non_missing_events=None,
                 avg_prec=None, f1_0=None, f1_1=None, avg_f1=None):
        """
        Instantiate NearestNeighbor object.
        :param data: A pd.Dataframe with ideally 3 columns (lon, lat, target
        :param neighbors: Number of neighbors to use for prediction
        :param how: The default is 'mode'-most occuring
        :param target: The target variable to predict
        :param time_window: The default is 7 days, defines a time window over which we can pick data
        """
        self.model_type = model_type
        self.model_name = model_name
        self.box_id = box_id
        self.psu = None
        self.region = None
        self.num_non_missing_events=non_missing_events
        self.total_non_missing_events = total_non_missing_events
        self.num_missing_events = missing_events
        self.num_invalid_events = invalid_events
        self.support = support
        self.tot_pred_0 = tot_pred_0
        self.tot_pred_1 = tot_pred_1
        self.correct_pred_0 = correct_0
        self.correct_pred_1 = correct_1
        self.actual_0 = actual_0
        self.actual_1 = actual_1
        self.accuracy = accuracy
        self.precision_0 = precision_0
        self.precision_1 = precision_1
        self.avg_precision = avg_prec
        self.recall_0 = recall_0
        self.recall_1 = recall_1
        self.window_length = window_len
        self.f1_score_0 = f1_0
        self.f1_score_1 = f1_1
        self.avg_f1_score = avg_f1
        self.neighbors = neighbors
        self.max_training_cases = max_training_cases
        self.min_training_cases = min_training_cases
        self.mean_training_cases = mean_training_cases
        self.median_training_cases = median_training_cases
        self.precision_0_sc = None
        self.precision_1_sc = None
        self.avg_precision_sc_macro = None
        self.avg_precision_sc_micro = None
        self.recall_0_sc = None
        self.recall_1_sc = None
        self.avg_recall_sc_macro = None
        self.avg_recall_sc_micro = None
        self.f1_score_0_sc = None
        self.f1_score_1_sc = None
        self.avg_f1_score_sc_macro = None
        self.avg_f1_score_sc_micro = None


class ImputationNearestNeighbor:
    """
    Nearest neighbor models
    """
    BOX_METADATA = None

    # in cases where we are training on all data, we persist the model to save time
    CLF = None

    def __init__(self, data='X', neighbors=1, target='event_type_str', how='frequent',
                 time_window=7, direction='both', out_of_box_model = None, pred_features = None):
        """
        Instantiate NearestNeighbor object.
        :param data: A pd.Dataframe with ideally 3 columns (lon, lat, target
        :param neighbors: Number of neighbors to use for prediction
        :param how: The default is 'mode'-most occuring
        :param target: The target variable to predict
        :param time_window: The default is 7 days, defines a time window over which we can pick data
        """
        self.neighbors = neighbors
        self.how = how
        self.data = data
        self.target_var = target
        self.direction = direction
        self.time_window = time_window
        self.out_of_box_model = out_of_box_model
        self.prediction_features = pred_features
        self.num_training_examples = None

    def generate_box_metadata(self, box_file):
        """
        Returns a data frame with [box_id, lon, lat solely for distance calculations
        :return: pd.Dataframe
        """
        df = pd.read_csv(box_file, usecols=['LONG', 'LAT', 'ClusterId', 'BoxID'])
        df.rename(columns={'LONG': 'lon', 'LAT': 'lat','ClusterId':'psu','BoxID':'box_id'}, inplace=True)
        df = df.sort_values(by='psu')

        self.BOX_METADATA = df

    def generate_train_data(self, raw_data=None, target_date=None, boxes = None):
        """
        Selects only events within two time bounds to use as trainign set
        :param data: The whole dataset to select from (probably pandas dataframe)
        :param target_date: The date we want to predict on
        :return: A pd.Dataframe
        """
        # return all data if time window is -1-we use all data
        if self.time_window == -1:
            return raw_data

        delta = timedelta(days=self.time_window)
        start = target_date-delta
        end = target_date+delta
        mask = (raw_data['datetime_sent_hr'] >= start) & (raw_data['datetime_sent_hr'] <= end)

        # keep only the boxes under consideration
        train_data = raw_data[raw_data['box_id'].isin(boxes)]

        # and for these boxes, keep only data in specified time-window
        train_data2 = train_data.loc[mask]

        return train_data2

    def generate_event_freqs(self, df=None, by_box=False, cond=None):
        """
        Returns a dictionary object of frequency of each event by the hour.
        :param df: A dataframe with events in a time period of interest(e.g, 7 day window)
        :param by_box: if frequencies have to computes by box, otherwise, pools togather all events and computes
        :param cond:
        :return:
        """
        target = self.target_var
        if by_box:
            # Add hour column
            df['hr'] = df['datetime_sent_hr'].apply(lambda x: x.hour)

        else:
            hr_cnts = df.groupby(['hour_sent', target])[target].agg(['count'])

            hr_cnts = hr_cnts.reset_index()

        return hr_cnts

    def get_neighbors(self, box_id= 1, loc=None):
        # ---------GET NEIGHBORS-----------------------------
        # remove box id under consideration so that the idea of neighbors makes sense
        if self.neighbors == 0:
            return [box_id]

        bx = self.BOX_METADATA[self.BOX_METADATA.box_id != box_id]

        bx.is_copy = False

        bx['dist'] = bx.apply(lambda row: ut.calculate_distance([row['lat'], row['lon']],
                                                                loc), axis=1)

        nearest_n = bx.sort_values(by=['dist'], ascending=True)[:self.neighbors]

        neighbors = list(nearest_n.box_id.values)

        neighbors.append(box_id)  # since we also want to learn from the same box

        return neighbors

    def predict_with_majority_classifier(self, train_data = None):
        """
        Returns class with majority in training dataset
        :param train_data:
        :return:
        """
        cnts = train_data[self.target_var].value_counts()
        cnts_dict = cnts.to_dict()
        max_class = max(cnts_dict.items(), key=operator.itemgetter(1))[0]

        return max_class

    def predict_with_random_classifier(self, train_data = None):
        """
        Pick one class randomly
        :param train_data:
        :return:
        """
        classes = list(train_data[self.target_var].unique())
        return random.choice(classes)

    def predict_with_nearest_neighbor(self, train_df = None, by_box=None, prediction_date=None):

        # ---------GENERATE EVENT FREQUENCIES--------------------
        event_freqs = self.generate_event_freqs(df=train_df, by_box=False)

        # ---------RETRIEVE VALUE FOR THAT HOUR------------------
        pred_hr = prediction_date.hour

        events_hr = event_freqs[event_freqs.hour_sent == pred_hr]

        predicted_event = events_hr.max(axis=0)[self.target_var]

        return predicted_event

    def predict(self,target_loc=None, prediction_date=None, data=None, box_id=100, model_type='nn', test_X=None):
        """
        Returns prediction based on previous events in a time window. By default, events include those from
        the same box BUT we add events form neighboring boxes
        :param target_loc: loc to be predicted
        :param prediction_date: date of prediction
        :param data: The data to learn from
        :param box_id: Box-ID of box under consideration
        :return: An event_type_str
        """

        # ----------GET NEIGHBORS---------------------------
        neighbors = self.get_neighbors(box_id=box_id, loc=target_loc)

        # ----------GET TRAINING DATA-----------------------
        if not data:
            data = self.data

        if self.neighbors == -1 and self.time_window == -1:
            train_df = data
        else:
            train_df = self.generate_train_data(target_date=prediction_date, raw_data=data, boxes=neighbors)

        self.num_training_examples = train_df.shape[0]

        # ----------PICK MODEL-------------------------------
        if model_type == 'nn':
            pred = self.predict_with_nearest_neighbor(train_df=train_df, prediction_date=prediction_date)
            return pred
        elif model_type == 'out':
            trained_model = self.train_out_of_box_model(train=train_df)
            predicted = self.predict_with_out_of_the_box_model(X=test_X, clf=trained_model)
            return predicted[0]
        elif model_type == 'rand':
            return self.predict_with_random_classifier(train_data=train_df)
        elif model_type == 'major':
            return self.predict_with_majority_classifier(train_data=train_df)

    def train_out_of_box_model(self, train=None):
        """
        For the sake of evaluation, train an out the box model wih similar conditions as nearest neighbor
        :param train:
        :return:
        """
        # ----------IF MODEL ARLEADY EXIST, CASE OF NEIGHBORS = -1--
        if self.neighbors == -1 and self.time_window == -1:
            if self.CLF:
                return self.CLF
            else:
                # ----------INITIALISE MODEL---------------------------
                clf = self.out_of_box_model

                # ----------FIT MODEL---------------------------------
                X = train[self.prediction_features].values
                y = train[self.target_var]
                clf.fit(X,y)
                self.CLF = clf
                return clf
        else:
            # ----------INITIALISE MODEL---------------------------
            clf = self.out_of_box_model

            # ----------FIT MODEL---------------------------------
            X = train[self.prediction_features].values
            y = train[self.target_var]
            clf.fit(X, y)
            return clf

    def predict_with_out_of_the_box_model(self, X=None, clf=None):
        """
        Prediction for a single test instance using out of the box model
        :param X:
        :return:
        """
        predicted = clf.predict(X=X)

        return predicted