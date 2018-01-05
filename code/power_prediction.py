#Import packages
import pytz
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
import re
from IPython.display import Image
import itertools
from math import sin, cos, sqrt, atan2, radians
from random import shuffle
from random import randint


def generate_train_data(d_raw, date, num_days, target):
    """
    This function returns a training dataset for prediction.

    @param d_raw: reasonably preprocessed data
    @param date: date threshold (when to stop) e.g., 2016/10/20
    @param num_days: Number of days to consider
    @return: Returns a dataset where each row represents a time interval(num_days) with corresponding
    @raise keyError: raises an exception
    """

    # create a list of num_days time intervals covering 2016 october-1 up date
    start = d_raw.date_powerfailure.min().date()
    end = datetime.strptime(date, '%Y/%m/%d').date()
    delta = timedelta(days=num_days)
    date_list = []

    while start <= end:
        tup = (start, start + delta)
        date_list.append(tup)

        start += timedelta(days=num_days + 1)

    # Use cartesian product to create a dataframe of the dates created above
    # and boxID's
    date_boxid = pd.DataFrame(list(itertools.product(date_list, list(d_raw.BoxID.unique()))),
                              columns=['dates', 'BoxID'])

    result = []
    d_10dy = pd.DataFrame
    for date in list(date_list):
        # Select events in this range
        mask = (d_raw['date_powerfailure'] >= date[0]) & (d_raw['date_powerfailure'] <= date[1])
        d_raw_within = d_raw.loc[mask]

        # Summarise powerouts to num_days level
        d_raw_by_box = pd.DataFrame(d_raw_within.groupby(['BoxID'])[target].max())
        d_raw_by_box.reset_index(inplace=True)

        new_powerout_col = 'pwrout_' + str(num_days) + 'dy'
        d_raw_by_box.rename(columns={target: new_powerout_col}, inplace=True)
        d_raw_by_box['dates'] = [date for i in range(d_raw_by_box.shape[0])]

        # create a df of summarised power failure events
        if len(result) == 0:
            d_10dy = d_raw_by_box
        else:
            d_10dy = d_10dy.append(d_raw_by_box)

        result.append(d_raw_by_box)

    # print (date_boxid.head())
    d2 = pd.merge(left=bx, right=d_10dy, on='BoxID', how='right')

    # Rearrange columns
    d2 = d2[['dates', 'ClusterId', 'BoxID', 'LONG', 'LAT', 'pwrout_10dy']]

    # rename soem cols
    d2.rename(columns={'LONG': 'lon', 'LAT': 'lat'}, inplace=True)
    return d2


def calculate_distance(pt1, pt2):
    """
    Computes distance between two geographic coordinates
    :param pt1: [Lat,Lon] for first point
    :param pt2: [Lat,Lon] for second
    :returns distance in km between the two points
    """
    # Radius of the earth in km (Hayford-Ellipsoid)
    EARTH_RADIUS = 6378388 / 1000

    d_lat = radians(pt1[0] - pt2[0])
    d_lon = radians(pt1[1] - pt2[1])

    lat1 = radians(pt1[0])
    lat2 = radians(pt2[0])

    a = sin(d_lat / 2) * sin(d_lat / 2) + \
        sin(d_lon / 2) * sin(d_lon / 2) * cos(lat1) * cos(lat2)

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return c * EARTH_RADIUS


def predict_with_nearest_neighbor(poi, points, target):
    """
    Computes distance between a point of interest and the rest of the points
    :param pt1: [Lat,Lon] for first point
    :param pt2: [Lat,Lon] for second
    :returns a dict object with BoxID and distances
    """
    # dict object to
    distances = {}
    min = 500000.000
    min_bx = ''
    predicted_pwrstate = ''

    for bx in list(points.BoxID.unique()):
        d_bx = points[points.BoxID == bx]
        dist = calculate_distance(poi, [d_bx.lat, d_bx.lon])

        if dist < min:
            min = dist
            predicted_pwrstate = d_bx[target]
            min_bx = bx

    return [min_bx, predicted_pwrstate, min]


def evaluate_nearest_neighbor_predictor(df, k, num_days):
    """
     This function quickly evaluates perfomance of nearest neighbor approach
     by leaving out the location being tested.
    :param df: Data under consideration
    :param k: NUmber of locations to test for
    :param num_days: In this case, since they are many dates, how many dates to test
    :returns a big dict object
    """
    # Get a list of boxes/locations to predict and shuffle them
    boxes = list(df.BoxID.unique())

    shuffle(boxes)

    all_res = []
    for k_i in range(k):
        # Randomly select location/box to predict
        box_id = boxes[randint(0, k)]

        # The actual location
        df_bx = df[df.BoxID == box_id]

        xy = [df_bx.iloc[0].lat, df_bx.iloc[0].lon]

        # remove all data with this location
        train = df[df.BoxID != box_id]

        # Get correct label information
        test = df[df.BoxID == box_id]

        # Available dates on which we can test our predictor and shuffle them
        dates = [list(test.date_powerfailure.unique())[i] for i in range(num_days)]

        shuffle(dates)

        res = {}
        tot = 0
        correct = 0
        for d in dates:
            # Get a subset of events for this date
            df_d = train[train.date_powerfailure == d]

            # Make prediction
            predicted = predict_with_nearest_neighbor(xy, df_d, 'pwrout_1dy')
            print(predicted)
            # Actual
            actual = test[test.date_powerfailure == d].pwrout_1dy

            res[box_id] = {'date': d, 'actual': actual, 'predicted': predicted[1], 'dist': predicted[2]}

            all_res.append(res)

            if actual == predicted[1]:
                correct += 1

            tot += 1
        # compute accuracy for this Location
        print('Accuracy %s ' % (correct / tot * 100))




#Set up Working Directory and Data Files
drive = 'Mac'
if drive == 'G':
    data_dir = '/Users/dmatekenya/Google\ Drive/World-Bank/electricity_monitoring/01.data/'
elif drive == 'Mac':
    data_dir = '/Users/dmatekenya/The_Bank/electricity_monitoring/01.data/'
else:
    data_dir = '..\\01.data\\'


#Load data of hourly power events: data prpcessed by JP
d = pd.read_csv(data_dir + 'powerout.csv')

#Because I want to drop all rows with check less than 2
d.ix[d.checks==-9, ['checks']] = 9

size = d.shape[0]

d = d[d.checks >= 2]

print ("%s rows dropped...."%(size-d.shape[0]))


#Kepp only required columns
to_keep = ['BoxID', 'date_powerfailure', 'date_powerfailure_hour',
       'date_powerback', 'date_powerback_hour','duration','POWERout']
d = d[to_keep]

#change to datetime objects
d['date_powerfailure'] = d.apply(lambda x: pd.datetime.strptime(x['date_powerfailure'], '%d%b%Y'), axis=1)

#d['date_powerback'] = d[].apply(lambda x: pd.datetime.strptime(x['date_powerback'], '%d%b%Y'), axis=1)


#Collapse data to daily level
#For all the hours of the day, I simply take the maximum POWERout val
d_dy = pd.DataFrame(d.groupby(['BoxID','date_powerfailure'])['POWERout'].max())
d_dy.reset_index(inplace=True)
d_dy.rename(columns={'POWERout': 'pwrout_1dy'},inplace=True)


#Evaluate predictpr
 #Get box location
bx = pd.read_csv(data_dir + 'Boxes.csv', usecols=['LONG','LAT','ClusterId','BoxID'])
d2 = pd.merge (left = bx, right=d_dy,on='BoxID', how='right')
#rename soem cols
d2.rename (columns={'LONG': 'lon','LAT':'lat'},inplace=True)

evaluate_nearest_neighbor_predictor(d2, 1, 5)