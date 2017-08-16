"""
For making out of sample predictions and imputations
"""
from datetime import timedelta

import pandas as pd

from pypower import data_utils as ut


class ImputationNearestNeighbor:
    """
    Nearest neighbor models
    """
    BOX_METADATA = None

    def __init__(self, data='X', neighbors=1, target='event_type_str', how='frequent',
                 time_window=7, direction='both'):
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
        if by_box:
            # Add hour column
            df['hr'] = df['datetime_sent_hr'].apply(lambda x: x.hour)

        else:
            hr_cnts = df.groupby(['hour_sent', 'event_type_str'])['event_type_str'].agg(['count'])

            hr_cnts = hr_cnts.reset_index()

        return hr_cnts

    def predict(self,target_loc=None, prediction_date=None, data=None, box_id=100):
        """
        Returns prediction based on previous events in a time window. By default, events include those from
        the same box BUT we add events form neighboring boxes
        :param target_loc: loc to be predicted
        :param prediction_date: date of prediction
        :param data: The data to learn from
        :param box_id: Box-ID of box under consideration
        :return: An event_type_str
        """

        # ---------GET NEIGHBORS-----------------------------
        # remove box id under consideration so that the idea of neighbors makes sense
        bx = self.BOX_METADATA[self.BOX_METADATA.box_id != box_id]
        bx.is_copy = False

        bx['dist'] = bx.apply(lambda row: ut.calculate_distance([row['lat'], row['lon']],
                                                                              target_loc), axis=1)

        nearest_n = bx.sort_values(by=['dist'], ascending=True)[:self.neighbors]

        neighbors = list(nearest_n.box_id.values)

        neighbors.append(box_id) # since we also want to learn from the same box

        # ----------GET TRAINING DATA---------------------------
        if not data:
            data = self.data

        train_df = self.generate_train_data(target_date=prediction_date, raw_data=data, boxes=neighbors)


        # ---------GENERATE EVENT FREQUENCIES--------------------
        event_freqs = self.generate_event_freqs(df=train_df, by_box=False)

        # ---------RETRIEVE VALUE FOR THAT HOUR------------------
        pred_hr = prediction_date.hour

        events_hr = event_freqs[event_freqs.hour_sent == pred_hr]

        predicted_event = events_hr.max(axis=0)[self.target_var]

        return predicted_event
