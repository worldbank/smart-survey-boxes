# -*- coding: utf-8 -*-
#Import packages
import os
import pytz
import pandas as pd
import numpy as np
import random
from math import sin, cos, sqrt, atan2, radians
from datetime import datetime, date, time, timedelta
from math import sin, cos, sqrt, atan2, radians
import re
import itertools
import pandas as pd

from flask import Flask, jsonify, request, render_template
from flask_debugtoolbar import DebugToolbarExtension
import json
import geojson


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
df_avg_dur = ' '

def preprocess_powerout_data (file_name):
    """
    Does some data wrangling for use in the prediction of average duration
    :param file_name: csv file of power outages processed by JP
    :returns a pd.DataFrame object
    """
    #Load data of hourly power events: data prpcessed by JP
    df = pd.read_csv(file_name)

    #Kepp only required columns
    to_keep = ['BoxID', 'date_powerfailure', 'date_powerfailure_hour',
       'date_powerback', 'date_powerback_hour','duration','POWERout']

    d = df[to_keep]

    #Change duration to hours
    d['duration_hrs'] = d['duration']/60

    #consider only valid power out events
    d = d[d['duration_hrs'].notnull()]
    
    #change to datetime objects
    d['date_powerfailure2'] = d.apply(lambda x: pd.datetime.strptime(x['date_powerfailure'], '%d%b%Y'), axis=1)

    d.drop('date_powerfailure', axis=1, inplace=True)

    d.rename (columns={'date_powerfailure2': 'date_powerfailure'},inplace=True)

    return d

def k_nearest_neighbors(df, target, num_neighbors=5, how='wmean'):
    """
    Given a dataframe with lat/lon and average duration of power failure for that location
    this function finds the nearest point to the target point and return the avg power duration
    from the nearest point.
    :param df: pd.DataFrame with cols: BoxID, lat/lon, avg power failure duration (hours) 
    :param target: [lat,lon]-The unsampled location where the avg power failure duration needs to be estimated
    :param num_neighbors: NUmber of neighbors to use for the nearest neigbor
    :param how: The approach to use. If num_neighbors == 1, just return value of nearest point..otherwise, it can
    be 'mean', 'mode' or 'median'.
    :returns the avg power failure duration and posssibly the BoxID
    """
    #Just incase, remove all duplicate lat/lon pairs
    df = df.drop_duplicates (['lon','lat'])
    
    df ['dist'] = df.apply(lambda row: calculate_distance([row['lat'],row['lon']],target),axis=1)
    
    #In the simplest case, assign to this new location valaue based on a single nearest neighbor
    if num_neighbors==1:
        min_dist = df.dist.min()
        predicted = df[df.dist==min_dist].duration_hrs.values[0]
        return predicted
    else: #cases with more than one neighbor
        #Select only the 'num_neaighbors' nearest neighbors
        df = df.sort_values (by = ['dist'], ascending=True)[:num_neighbors]
        if how=='wmean':
            #Add weight based on distance
            df['dist_wght'] = 1/df['dist']
            
            #compute simple weighted mean
            wmean = (df['duration_hrs'] * df['dist_wght']).sum() /df.dist_wght.sum()
            
            return wmean
        elif how == 'mean':
            return df['duration_hrs'].mean (axis=0)
        elif how == 'median':
            return df['duration_hrs'].median (axis=0)

def compute_mean_duration (df, cut_off_date):
    '''
    The idea is to track number of hours when power is off not
    later than some given cut off date by box.
    @param:  df: the data file under consideration
    @param:  cut_off_date: upper date limit for considerin events
    @return: a pd.DataFrame object containing average duration of each box
    '''
    #select events earlier than cut off date
    d_date = df[df.date_powerfailure <= cut_off_date]
    
    #select only powerout events
    d_out = d_date[d_date.duration_hrs.notnull()]
    
    #Exclude all outages longer than 1 day
    print ('Leaving out %s events whose power out duration is over 24 hrs!!!' %(len(d_out[d_out.duration_hrs>=24])))
    
    d_out = d_out[d_out.duration_hrs < 24]
    
    #Group by box id and compute mean
    by_box = d_out.groupby(['BoxID'])['duration_hrs'].mean()
    
    print ('Completed adding mean to df.....')
    return by_box.reset_index()

def data2geojson(df):
    features = []
    df.apply(lambda X: features.append( 
            geojson.Feature(geometry=geojson.Point((X["lat"], X["lon"])))), axis=1)
    
    return features

def calculate_distance (pt1,pt2):
    """
    Computes distance between two geographic coordinates
    :param pt1: [Lat,Lon] for first point
    :param pt2: [Lat,Lon] for second
    :returns distance in km between the two points
    """
    # Radius of the earth in km (Hayford-Ellipsoid)
    EARTH_RADIUS = 6378388/1000
    
    d_lat = radians (pt1[0] - pt2[0])
    d_lon = radians (pt1[1] - pt2[1])
    
    lat1 = radians(pt1[0])
    lat2 = radians(pt2[0])
    
    a = sin(d_lat/2) * sin(d_lat/2) + \
        sin(d_lon/2) * sin(d_lon/2) * cos(lat1) * cos(lat2)
        
    c = 2 * atan2(sqrt(a), sqrt(1-a))

    return c * EARTH_RADIUS
    
def predict_with_nearest_neighbor (poi,points,target):
    """
    Computes distance between a point of interest and the rest of the points
    This predicts whether there was a power outage 
    :param poi: [Lat,Lon] for the location where we want to make the prediction
    :param points: the training data
    :param target: target variable for prediction (e.g., 1 dy)
    :returns a dict object with BoxID and distances
    """
    #dict object to 
    distances = {}
    min = 500000.000
    min_bx = ''
    predicted_pwrstate = ''
    
    for bx in list(points.BoxID.unique()):
        d_bx = points[points.BoxID == bx]
        dist = calculate_distance (poi,[d_bx.lat,d_bx.lon])
        
        if dist < min:
            min = dist
            predicted_pwrstate = d_bx[target].values[0]
            min_bx = bx
    
    #print (predicted_pwrstate)
    return [min_bx, predicted_pwrstate, min]

app = Flask(__name__)
app.secret_key = 'development key'

toolbar = DebugToolbarExtension(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/click_loc", methods = ['GET','POST'])
def predict_power_states():
    if request.method == "POST":
        data = request.data
        clicked = json.loads (data.decode())
        xy = [clicked['lat'],clicked['lon']]
        
        #--------------------------------------------------
        #PREDICT IF THERE WAS POWER OUT IN PAS N-DAYS
        #--------------------------------------------------
        df = pd.read_csv(APP_ROOT + '/input/powerouts_1dy.csv')
        dates = list(df.date_powerfailure.unique())
        #Pick a random date-later a user could choose the date
        pred_date = dates[random.randint(0,len(dates))-1]
        pred_date = pd.Timestamp(pred_date)
        print ('Date-Type : %s' %(type(pred_date)))
        df_d = df[df.date_powerfailure==pred_date]
        predicted = predict_with_nearest_neighbor (xy, df_d, 'pwrout_1dy')[1]
        state = ''
        if predicted == 1:
            state = 'Yes'
        else:
            state = 'Nope'

        #--------------------------------------------------
        #PREDICT AVERAGE POWER OUT DURATION (HRS)
        #--------------------------------------------------
        #import box details
        bx = pd.read_csv(APP_ROOT + '/input/Boxes.csv', usecols=['LONG','LAT','ClusterId','BoxID'])
        #rename some cols
        bx.rename (columns={'LONG': 'lon','LAT':'lat'},inplace=True)

        #compute average power out duration based on randomly picked date (pred_date selected above)
        df_avg = compute_mean_duration (df_avg_dur, pred_date)

        d = pd.merge (left = df_avg, right=bx, on='BoxID', how='left')

        #Make the prediction using target = xy, 
        pred_5_wmean = k_nearest_neighbors(d, xy, 5,'wmean')
        
        print ('Avg Dur ', pred_5_wmean)

        return jsonify([pred_date,state, pred_5_wmean])
    else:
        return render_template("index.html")


@app.route("/boxes")
def add_boxes_location():
    """Get location of the boxes and them to the map"""
    df = pd.read_csv(APP_ROOT + '/input/powerouts_1dy.csv')
    df.x = df.lat.unique()
    df.y = df.lon.unique()
    xy = pd.DataFrame({'lat':df.x,'lon':df.y})
    pts = data2geojson(xy)
    return jsonify(pts)

if __name__ == "__main__":
    #Preprocess this data to get it ready for when its needed
    pwrout_file = APP_ROOT+ '/input/powerout.csv'
    
    df_avg_dur = preprocess_powerout_data (pwrout_file)
    
    app.run(host='0.0.0.0',port=5000,debug=True)
    
    