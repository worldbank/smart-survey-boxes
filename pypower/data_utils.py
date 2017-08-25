"""
A helper module for preprocessing data to various formats before being used.
"""

import csv
import re

import sys
import traceback
import xml.etree.ElementTree as ET
from collections import OrderedDict
from datetime import datetime, timedelta
from math import radians, cos, sin, atan2, sqrt

import pandas as pd
from pytz import timezone as tz


class Box:
    """
    Holds information about each box and the corresponding data about the box.
    """

    TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

    def __init__(self, box_id=1024, psu=1, lon=1, lat=2, date_collection_started=None):
        self.box_id = box_id
        self.psu = psu
        self.lon = lon
        self.lat = lat
        self.date_collection_started = date_collection_started
        self.urban_rural = None
        self.region = None
        self.district = None
        self.actual_events = []  # Storing in a set to ensure there are no duplicate events for the same box
        self.hourly_events = None  # Stored in a dictionary for fast access
        self.daily_summaries = OrderedDict()  # Store summariesd variables for each day

    def drop_events(self):
        """
        Drops test events and those before data collection started
        :return:
        """
        # print('Number of actual events....{}'.format(len(self.actual_events)))
        new_events = []

        for e in self.actual_events:
            if e.before_date_collection:
                continue

            if e.event_type_str == 'test':
                continue

            new_events.append(e)

        self.actual_events = new_events
        # print('Number of actual events after dropping events before date of collection....{}'
        #       .format(len(self.actual_events)))

    def add_event(self, event):
        """
        The reason for creating a special set method is to first check
        if this event happened after date-collection started
        :param: event: the event we want to add
        :return: Doesnt return anything
        """

        event_date = event.datetime_sent
        event_date_str = event_date.strftime(self.TIME_FORMAT)
        collection_date_str = self.date_collection_started.strftime(self.TIME_FORMAT)

        if event.datetime_sent <= self.date_collection_started:
            # print('Event date {} is before date of collection: {} for box {}....'
            #       .format(event_date_str, collection_date_str, self.box_id))
            event.before_date_collection = True

        self.actual_events.append(event)

    def get_box_metadata(self, required_box_metadata=None):
        """
        A helper function to get get a few selected metadata about a box
        :param required_box_metadata: List of required metadata for this box
        :return: A dict object of box attribute and its value
        """
        all_box_metadata = self.__dict__
        required_box_metadata_dict = {}

        if not required_box_metadata:
            required_box_metadata = ['box_id', 'psu', 'lon', 'lat']

        for i in required_box_metadata:
            if i in all_box_metadata:
                required_box_metadata_dict[i] = all_box_metadata.get(i)
            else:
                print('Requested attribute not available')

        return required_box_metadata_dict

    @classmethod
    def save_observed_events_as_sms_v1(cls, boxes_data, box_attr=None, event_attr=None,
                                       output_file="/Users/dmatekenya/sms_v1.csv"):
        """
         Saves sms_'date'_v1 as explained in the workflow.
        :param: output_file: The file where to save the data
        :param: boxes_data: A list of Box objects with events
        :return:
        """
        if not event_attr:
            event_attr = ['datetime_sent_raw', 'datetime_received', 'datetime_sent', 'message',
                          'event_type_str', 'event_type_num', 'box_state']
        if not box_attr:
            box_attr = ['box_id', 'psu', 'loc']

        header = box_attr + event_attr
        try:
            with open(output_file, 'w', encoding='UTF-8') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()

                for bx in boxes_data:
                    box_details = bx.get_box_metadata(box_attr)

                    for item in bx.actual_events():
                        event_metadata = item.get_selected_event_metadata(event_attr)
                        event_metadata.update(box_details)
                        writer.writerow(event_metadata)
        except Exception as e:
            print('Failed to write to file %s' % e)

    def count_messages(self, which='all'):
        """
        Returns message count per day
        :param which: whether to use all messages or only ping messages
        :return:
        """

        # Get hourly events
        hr_events = self.hourly_events

        # create a dataframe
        df = pd.Data(hr_events)

        if which == 'all':
            mask = df[df['event_type_str'] != 'not_observed']

            # summarise daily
            df.ix['msg', mask] = df.apply(lambda x: 1, axis=1)

            df_day = df.groupby(['date_sent'])['msg'].count()

            return df_day
        else:
            df_day = df.groupby(['date_sent'])['ping_msg'].count()

            return df_day

    def count_power_fails(self, time_interval):
        """
        This function computes number of outages per unit of time
        for all the events which happened when the boxes were functioning.
        :param time_intervals: e.g. daily
        :returns a pd.DataFrame object with the following columns: BoxID, lat/lon, time_unit, count
        """
        # Get hourly events
        hr_events = self.hourly_events

        # create a dataframe
        df = pd.Data(hr_events)

        if time_interval == 'daily':
            df_grped = df.groupby(['date_sen'])['power_state'].count()

            df_dy = df_grped.reset_index()

            return df_dy

    # TODO power mean duration

    def compute_mean_duration(df, df_bx, cut_off_date):
        '''
        The idea is to track number of hours when power is off not
        later than some given cut off date by box.
        @param:  df: the data file under consideration
        @param:  cut_off_date: upper date limit for considerin events
        @return: a pd.DataFrame object containing average duration of each box
        '''
        some_df = ''

        return some_df

    # TODO a single function which can generate variables for this box
    def generate_daily_summaries(self, variables=None):
        """
            For each date, compute some key variables as provided in the list
            ALL_MESSAGE_COUNT-Number of total messages received per day-potentially useful for monitoring box life
            PING_MESSAGE_COUNT-Number of total messages received per day-potentially useful for monitoring box life
            POWER_FAILURE_COUNT-Number of power failure occurence in that day'
            POWER_ON_HOURS-Number of hours power was on
            POWER_FAILURE_DURATION-Exact duration of power outages for this day, otherwise could just be complement of above
        :param variables: e.g., number of hours with power
        :return: Probably a dict object with the data
        """

        if not variables:
            variables = ['all_message_count', 'ping_message_count', 'power_failure_count',
                         'power_on_hours', 'power_failure_duration']

        return self.daily_summaries

    # TODO power_on_hours
    def power_on_hours(self):
        """
        For a single day, calculates how many hours power was on
        :return:
        """

    def generate_hourly_events(self, time_resolution='hr',
                                     after_event_threshold=12, before_event_threshold=12, how_to_insert='prev'):
        """
        Creates rectangular dataset at 1-hour intervals
        :param time_resolution: e.g., 1 hr
        :return:
        """
        # SET SOME THRESHOLD VALUES
        TIME_NTERVAL = 1

        # DROP TEST EVENTS FIRST AND THOSE BEFORE DATE OF COLLECTION
        self.drop_events()

        # get start and end end date of actual events
        sorted_events = sorted(list(self.actual_events), key=lambda x: x.datetime_sent_hr)

        # store actual events to this dict object
        all_events = {}
        for eve in sorted_events:
            all_events[eve.datetime_sent_hr] = eve

        # Loop through events 2 at a time and create blank events, populate the all_events
        for first, second in zip(sorted_events, sorted_events[1:]):
            generated_events = self.generate_blank_events(first, second, 1, after_event_threshold,
                                                          before_event_threshold, how_to_insert)
            # update all_events whenever they are available
            if generated_events:
                all_events.update(generated_events)

        # set hourly events property to rect
        self.hourly_events = all_events

    def assign_event_type(self, prev_event, time_after_previous_event, next_event, time_before_next_event,
                          is_missing_threshold_lower=12, is_missing_threshold_upper=6, how='prev'):
        """
        Assign event_type_str for an inserted event
        :param prev_event: Previous event
        :param time_after_previous_event: Number of hours passed after previous event
        :param next_event: Next event
        :param time_before_next_event: Number of hours before next event
        :param  is_missing_threshold_lower: Threshold for flagging as missing with reference to previous event
        :param is_missing_threshold_upper: Threshold for flagging as missing with reference to next event
        :param how: use 'previous'-default event only or 'both'
        :return:
        """
        # SCENARIO 1-HOURS AFTER PREVIOUS EVENT IS WITHIN ACCEPTABLE WINDOW
        if time_after_previous_event <= is_missing_threshold_lower:
            return prev_event

        # SCENARIO 2-TIME INTERVAL IS GREATER THAN ACCEPTABLE WINDOW
        # CHOICE-1(how='both'): USE NEXT EVENT IF TIME BEFORE NEXT EVENT IS WITHIN ACCEPTABLE
        # WINDOW-I DONT REALLY LIKE THIS IDEA
        # CHOICE-1(how='prev'): LABEL ALL EVENTS AS MISSING
        if how == 'both':
            if time_before_next_event <= is_missing_threshold_upper:
                return next_event
            elif time_before_next_event > is_missing_threshold_upper:
                return 'missing'
        elif how == 'prev':
            return 'missing'


    def generate_blank_events(self, ev0, ev1, time_interval=1, is_missing_threshold_lower=12,
                              is_missing_threshold_upper=12, how='prev'):
        """
        Inserts events between two observed events, interpolate (add event_type_str) to inserted event
        EVENT_TYPE_STR INSERTION RULE:
        A blank event takes the value of the last event as long as the number of hours passed is less than given threshold
        :param time_interval: fixed to 1 hr
        :param ev0: start-event
        :param ev1: end-event
        :param is_missing_threshold_lower: When to flag blank event as missing based on time from last event
        :param is_missing_threshold_upper: When to flag blank event as missing based time to next event
        :param how: whether to insert wiht previous only or to use both
        :return: a dict of events
        """

        last_event_time = ev0.datetime_sent_hr
        next_event_time = ev1.datetime_sent_hr

        # Do nothing if time interval is 0
        if last_event_time == next_event_time:
            return

        # create events
        ts = last_event_time + timedelta(hours=time_interval)  # start an hour after actual event

        # dict to return
        events_dict = {}
        while ts < next_event_time:
            hrs_after_last_event = (ts - last_event_time).total_seconds() / 3600

            hrs_before_next_event = (next_event_time-ts).total_seconds() / 3600

            # create new event with message body set to 'not_observed'
            event_obj = Event(message='not_observed', datetime_sent_hr=ts, datetime_sent_raw=None)

            # set all important attributes
            event_obj.set_attributes()

            # to carry over last event type or not
            event_obj.event_type_str = self.assign_event_type(ev0.event_type_str, hrs_after_last_event,
                                                          ev1.event_type_str, hrs_before_next_event,
                                                          is_missing_threshold_lower,
                                                          is_missing_threshold_upper, how)

            event_obj.data_source = 'insertion'  # source of event data

            # set time since_last_event
            event_obj.hrs_since_event = hrs_after_last_event
            event_obj.hrs_before_event = hrs_before_next_event

            # set event_cate_num
            event_type = {'test': 0, 'pback': 1, 'pfail': 2, 'pon_mon': 3, 'pfail_mon': 4, 'missing': -1}
            event_obj.event_type_num = event_type.get(event_obj.event_type_str)

            # add event to dict
            events_dict[ts] = event_obj

            # increment datetime by 1 hr
            ts += timedelta(hours=time_interval)

        return events_dict

    def dataframe_from_actual_events(self, columns=None):
        """
        creates pandas dataframe from actual events after dropping events which ocurred before date of collection.
        :return: pandas dataframe
        """
        if not columns:
            columns = ['box_id', 'psu', 'lon', 'lat','datetime_sent_raw', 'str_datetime_sent',
                        'str_datetime_sent_hr', 'event_type_str', 'event_type_num', 'ping_event', 'data_source']

        # change from dictionary to list
        box_metadata = self.get_box_metadata()

        # DROP EVENTS WHICH OCCURRED BEFORE DATE OF COLLECTION
        self.drop_events()

        raw_events = self.actual_events
        actual_events_metadata = []

        try:
            for eve in raw_events:
                event_attr = eve.get_selected_event_metadata()
                event_attr.update(box_metadata)
                actual_events_metadata.append(event_attr)

            # create dataframe
            df = pd.DataFrame(actual_events_metadata)
            df.sort_values(by='str_datetime_sent', inplace=True)

            # reorder columns
            df = df[columns]
        except Exception as e:
            print('Box ID with issues....{}'.format(box_metadata.get('box_id')))
            print(e)

        return df

    def dataframe_from_hourly_events(self, columns=None):
        """
        creates pandas dataframe for convinient usage in other processes
        :return: pandas dataframe
        """
        if not columns:
            columns = ['box_id', 'psu', 'lon', 'lat', 'datetime_sent_raw', 'datetime_received', 'str_datetime_sent',
                       'str_datetime_sent_hr', 'day_sent', 'hour_sent', 'month_sent', 'wk_day_sent', 'wk_end',
                       'event_type_str', 'event_type_num', 'ping_event', 'data_source']

        # change from dictionary to list
        box_metadata = self.get_box_metadata()

        # add box details to each event
        hr_events_metadata = []

        try:
            for eve in self.hourly_events.values():
                event_attr = eve.get_selected_event_metadata()
                event_attr.update(box_metadata)
                hr_events_metadata.append(event_attr)

            # create dataframe
            df = pd.DataFrame(hr_events_metadata)
            df.sort_values(by='datetime_sent_hr', inplace=True)

            # reorder columns
            df = df[columns]
        except Exception as e:
            print('Box ID with issues....{}'.format(box_metadata.get('box_id')))
            print(df.columns)
            print(df.head())
            print(e)

        return df


class Event:
    """
    An event is a single sms and they are many fields describing an event.
    """
    TAJIK_HOLIDAYS = ['2016-11-06', '2017-01-02', '2017-03-21', '2017-03-22', '2017-03-23',
                      '2017-05-1', '2017-05-9', '2017-06-25', '2017-06-26', '2017-06-27',
                      '2017-09-01', '2017-09-11', '2017-11-06', '2017-03-24']

    # Constants for handling time stamps and conversions
    TIME_ZONE = tz('Asia/Dushanbe')  # using Tajikistan time zone
    TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
    REF_DATE = datetime(1960, 1, 1, hour=0, minute=0, second=0, tzinfo=TIME_ZONE)
    TIME_CONSTANT = 315637200000  # The constant 315637200000 is being added as some kind of offset based on stata code

    def __init__(self, datetime_received=None, datetime_sent_raw=None, datetime_sent_hr=None, message=None):
        """
        box-state (-1: unknown, 1: alive, 0: dead)-initialised to -1 for unknown status
        power-state (-1: unknown, 1: on, 0: fail)-initialised to -1 for unknown status
        data_source: A label to indicate source of data (e.g., if based on observed value, we say actual, otherwise it
        could be interpolated (between events based on a rule), imputed (if value was missing, we will impute)

        :param time_stamp: Time stamp of the event
        :param box_id: Box-ID
        :param message: This is extracted from  the body of the message
        """
        self.datetime_sent_raw = datetime_sent_raw  # numeric datetime_sent as given in xml file
        self.datetime_received = datetime_received  # same as readable_date
        self.message = message  # message body from sms, for inserted events, message = 'not_observed'
        self.datetime_sent = None
        self.str_datetime_sent = None
        self.datetime_sent_hr = datetime_sent_hr
        self.str_datetime_sent_hr = None
        self.before_date_collection = False  # for detecting events before start of collection
        self.day_sent = None
        self.date_sent = None
        self.hour_sent = None
        self.year_sent = None
        self.month_sent = None
        self.wk_day_sent = None
        self.wk_end = 0 # its not weekend unless otherwise as below
        self.holiday = 0
        self.hrs_since_event = None  # same as above but here any message counts (relaxed)
        self.hrs_before_event = None  # same as above but here any message counts (relaxed)
        self.is_missing = 0  # indicates whether event data is missing (0-not missing, 1-missing)
        self.is_missing_how = 'relaxed'  # how was it decided whether event is (relaxed or strict)
        self.event_type_str = None
        self.event_type_num = None
        self.ping_event = 0
        self.power_state = -1  # set default to unknown
        # three sources: observed_event (actual), inserted_event (interpolated between events), imputed_event
        #  (when event is missing)
        self.data_source = None

    def set_attributes(self):
        # --------DATETIME and associated variables which is used for all time-stamp purposes-------
        if self.datetime_sent_hr:
            self.datetime_sent = self.datetime_sent_hr
        else:
            self.datetime_sent = self.convert_datetime_sent(self.datetime_sent_raw)  # convert when given numerical date
            # datetime_sent truncated to hour for cases when we create from raw numerical datetime
            self.datetime_sent_hr = self.datetime_sent.replace(minute=0, second=0)

        self.str_datetime_sent = self.datetime_sent.strftime(self.TIME_FORMAT)
        self.str_datetime_sent_hr = self.datetime_sent_hr.strftime(self.TIME_FORMAT)
        self.day_sent = self.datetime_sent.day
        self.date_sent = self.datetime_sent.date()
        self.hour_sent = self.datetime_sent.hour
        self.year_sent = self.datetime_sent.year
        self.month_sent = self.datetime_sent.month
        self.wk_day_sent = self.datetime_sent.weekday()
        self.wk_end = 0  # its not weekend unless otherwise as below

        # whether event happened on weekend or not
        if self.wk_day_sent > 5:
            self.wk_end = 1

        self.holiday = self.is_holiday(self.date_sent)

        # --------VARIABLES TO BE USED FOR MISSING--------
        self.hrs_since_event = None  # same as above but here any message counts (relaxed)
        self.hrs_before_event = None  # same as above but here any message counts (relaxed)
        self.is_missing = 0  # indicates whether event data is missing (0-not missing, 1-missing)
        self.is_missing_how = 'relaxed'  # how was it decided whether event is (relaxed or strict)

        # --------TYPE OF EVENT VARIABLES--------
        event_type = {'test': 0, 'pback': 1, 'pfail': 2, 'pon_mon': 3, 'pfail_mon': 4, 'missing': -1}
        self.event_type_str = self.detect_event_type_from_message(self.message)  # category of event
        self.event_type_num = event_type.get(self.event_type_str)  # same as abobe but use numeric values

        # POWER STATE: unknown (-1), power on (1), power-off (0)
        power_state_dict = {'pback': 1, 'pfail': 0, 'pon_mon': 1, 'pfail_mon': 0, 'missing': -1}
        self.power_state = -1  # set default to unknown
        self.power_state = power_state_dict.get(self.event_type_str)

    def fix_datetime_received(self, x):
        """
        Fix readable date based on the fact that Russioan characters are being read like this:
        07 ???. 2016 ?. 18:29:39
        :return: Date String
        """
        date_str = self.datetime_received

        if len(date_str) == 24:
            corrected_date_str = '12/' + date_str[:2] + '/2016' + ' ' + self.datetime_received[-8:]

            return corrected_date_str
        elif len(self.datetime_received) == 23:
            corrected_date_str = '12/' + self.datetime_received[:2] + '/2016' + ' ' + '0' + self.datetime_received[-7:]

            return corrected_date_str

        return x

    def convert_datetime_sent(self, value):
        """
        Converts datetime_sent which comes as a float into a datetime object.
        :return: A timezone aware datetime object
        """
        new_date = self.REF_DATE + timedelta(milliseconds=float(value) + self.TIME_CONSTANT)

        return new_date

    def is_holiday(self, date_value):
        """
        Check if this event occurred on a holiday
        :return: 1 if its a holiday or
        """
        if date_value:
            date_str = date_value.strftime(self.TIME_FORMAT)

            if date_str in self.TAJIK_HOLIDAYS:
                return 1

        return 0

    def split_message(self):
        """
        For convineince of processing, splits message body into
        2 parts
        TODO: Find a better way to do this
        :param: idx index to indicate whethe to return first message or second one
        :return:
        """
        if "|" in self.message:
            msgs = self.message.split('|')
            return msgs
        elif ":" in self.message:
            msgs = self.message.split(':')
            return msgs
        else:
            return 'body'

    def detect_event_type_from_message(self, value):
        """
        Looks at the message and classifies it as follows:
        test-Messages with test in them
        pon_mon-power on monitoring
        pfail_mon-power off monitoring
        pfail-power fail
        :return: a string for the type of event
        """
        msg = self.message
        msg_2 = self.split_message()[1]

        event_type = ' '

        if msg[:4] == 'Test' or 'Test SMS' in msg:
            event_type = 'test'
        elif msg_2 == ' Monitoring... Power OK':
            event_type = 'pon_mon'
        elif msg_2 == ' Monitoring... Power Failure':
            event_type = 'pfail_mon'
        elif msg[:19] == 'Power Back detected' or msg[:10] == 'Power Back':
            event_type = 'pback'
        elif msg[:22] == 'Power Failure detected' or msg[:13] == 'Power Failure':
            event_type = 'pfail'
        else:
            event_type = 'None'

        return event_type

    def add_box_sate(self, type='relaxed'):
        """
        A way to determine whether box is alive or not.
        For relaxed, it means any event means the box is alive regardless of whether its a ping or not
        while 'strict' means we only use ping messages to determine box state
        :param type: "relaxed"
        :return: 1: box is alive or 0 when box isnt alive
        """
        event = self.event_type_str

        if type == 'relaxed':
            if event in ['test', 'pfail', 'pon_mon', 'pfail_mon']:
                return 1
        else:
            if event in ['pfail_mon', 'pfail_mon']:
                return 1

    def get_selected_event_metadata(self, required_event_metadata=None):
        """
        A helper function to retrieve a set of required metadata from all the attributes.
        :param required_event_metadata: List of required metadata
        :return: A dict object of event metadata with corresponding value
        """
        all_event_metadata = self.__dict__
        required_event_metadata_dict = {}

        if not required_event_metadata:
            required_event_metadata = ['datetime_sent_raw', 'datetime_received', 'str_datetime_sent', 'datetime_sent',
                                       'str_datetime_sent_hr', 'datetime_sent_hr', 'day_sent', 'date_sent', 'hour_sent',
                                       'month_sent','wk_day_sent', 'wk_end', 'holiday', 'hrs_since_event',
                                       'hrs_before_event', 'event_type_str','event_type_num','power_state', 'ping_event',
                                       'data_source']

        for i in required_event_metadata:
            if i in all_event_metadata:
                required_event_metadata_dict[i] = all_event_metadata.get(i)
            else:
                print('Requested attribute not available')

        return required_event_metadata_dict

    def set_time_elapsed_since_last_event(self, last_event_datetime):
        """
        returns  number of hours elapsed since last event for actual events
        :param: last_event_datetime: datetime for last observed event
        :return:
        """
        diff = last_event_datetime - self.datetime_sent_hr

        diff_hrs = diff.total_second() / 3600

        return diff_hrs


def convert_date_to_string(date_obj):
    time_format = '%Y-%m-%d %H:%M:%S'
    date_1_str = date_obj.strftime(time_format)

    return date_1_str


def convert_xml_to_csv(config=None, ts=None):
    """
     Converts xml_file to csv file
    :param config: Object with details about file location
    :return:
    """
    xml_file = config.get_xml_dir() + 'sms.xml'
    csv_file = config.get_raw_data_dir() + 'sms_' + ts.strftime('%m-%d-%Y') + '.csv'

    try:
        et = ET.parse(xml_file).getroot()
        lst = et.findall('sms')
        header = lst[0].keys()
    except Exception as e:
        print('Failed to read xml because of this error %s' % e)


    try:
        with open(csv_file, 'w', encoding='UTF-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            for item in lst:
                data = dict(item.items())
                writer.writerow(data)
    except Exception as e:
        print('Failed to write to file %s' % e)


def convert_to_aware_datetime_object(date_str):
    """
    A helper function to convert from string to
    :param date_str: A date string
    :return: A time zone aware time object
    """
    time_fomart = '%m/%d/%Y'
    time_zone = tz('Asia/Dushanbe')

    date_obj = datetime.strptime(date_str, time_fomart)
    date_obj = date_obj.replace(tzinfo=time_zone)

    return date_obj


def box_loc_metadata_as_dict(in_file):
    """
    Use pandas to read in 'Boxes.csv' which contain boxes details and output it as
    :param in_file: Boxes.csv-file which has box lat, lon and other metadata
    :return: a nested dict like so- {'box_id: {box metadata}}
    """

    bx = pd.read_csv(in_file,usecols=['ClusterId', 'DateCollectionStart', 'LONG', 'LAT', 'BoxID',
                                      'URB_RUR', 'REGION','DISTRICT'])

    bx.rename(columns={'ClusterId': 'psu', 'LONG': 'lon', 'LAT': 'lat', 'BoxID': 'box_id', 'REGION': 'region',
                       'DISTRICT': 'district'}, inplace=True)

    bx['DateCollectionStart'] = bx['DateCollectionStart'].apply(lambda x: convert_to_aware_datetime_object(x))

    # Convert Dataframe to dictionary-first make box_id
    bx.set_index('box_id', inplace=True)

    # Now convert to dict and return it

    return bx.to_dict(orient='index')


def event_dict_from_xml(xml_file):
    """
    Create a dict object with events and keyed by box-id
    :param xml_file: The sms.xml file
    :return: A dictionary of events indedxed by box id
    """
    try:
        e = ET.parse(xml_file).getroot()
        lst = e.findall('sms')
        lst_dict = [dict(elem.items()) for elem in lst]
        lst_dict_box = {}

        no_id_counter = 0
        for i in lst_dict:
            box_id = extract_box_id(i.get('body'))
            if box_id == 'no_box_id':
                # print('This event will be omitted because there is no box_id: %s'%(i.get('body')))
                no_id_counter += 1
                continue

            if int(box_id) in lst_dict_box:
                lst_dict_box[int(box_id)].append(i)
            else:
                lst_dict_box[int(box_id)] = [i]

        print('{} EVENTS OMITTED BECAUSE OF NO ID IN MESSAGE'.format(no_id_counter))
        return lst_dict_box
    except Exception as e:
        print('Failed to read xml because of this error %s' % e)


def create_box_obj_from_events(xml_file=None, box_metadata=None, after_event_threshold=12,
                               before_event_threshold=12,
                               how_to_insert='prev', debug_mode=True):
    """
    Creates box objects and events for the box. Also, computes all necessary properties for the box
    :param xml_file: raw sms.xml file
    :param box_metadata: Box details to get info such as lat/lon
    :param after_event_threshold: For use in inserting events
    :param before_event_threshold: For use in inserting events
    :param how_to_insert: Insertion approach
    :param debug_mode: Debug or not
    :return:
    """
    box_events = event_dict_from_xml(xml_file)  # convert sms.xml into dict for faster access (hopefully)
    box_objects = {}  # holds box objects which we will return

    i = 0 # for debugging
    for bx_id, value in box_metadata.items():

        box_obj = Box(date_collection_started=value.get('DateCollectionStart'),
                      box_id=bx_id, psu=value.get('psu'), lon=value.get('lon'),
                      lat=value.get('lat'))

        this_box_raw_events = box_events.get(bx_id)  # has raw event data

        # Generate event objects
        for item in this_box_raw_events:
            event = Event(datetime_received=item.get('readable_date'), datetime_sent_raw=item.get('date_sent'),
                          message=item.get('body'))

            # set all important attributes
            event.set_attributes()

            # indicate data source
            event.data_source = 'observed_event'

            #whether its ping event or not
            if event.event_type_str in ['pfail_mon', 'pon_mon']:
                event.ping_event = 1  # helper variable for counting events

            if event.event_type_str == 'None':
                #print('Impossible to determine event type: %s' % event.message)
                continue

            box_obj.add_event(event)

        # Clean up
        box_obj.drop_events()

        # check number of events-skip boxes with less than 5 messages
        if len(box_obj.actual_events) < 5:
            print('SKIPPING BOX {} BECAUSE IT HAS ONLY {} EVENTS..'.format(bx_id,len(box_obj.actual_events)))
            continue

        #Generate hourly events-could take long
        box_obj.generate_hourly_events('hr', after_event_threshold=after_event_threshold,
                                       before_event_threshold=before_event_threshold,
                                           how_to_insert=how_to_insert)

        # Add event object to Box object
        box_objects[bx_id] = box_obj

        i += 1
        # In debug mode only create 10 objects
        if debug_mode:
            if i == 10:
                break

    return box_objects


def extract_box_id(message):
    """
    Extracts Box-ID from the message body
    :return: BOX-ID
    """
    digits_in_msg = re.findall(r'\d+', message)
    if len(digits_in_msg) < 1:
        #print('It seems there is no box_id for this message--- %s' % message)
        return 'no_box_id'
    else:
        return digits_in_msg[0]


def save_datasets(box_objects=None, output_file_v1='sms_observed.csv', output_file_v2='sms_rect_hr.csv'):
    """
    Saves sms_todays date_v1.csv-the first processed file to be outputed.
    :param output_file: Full path of output file
    :param box_objects: Box objects to get data from
    :return: Saves to disk
    """
    try:
        for i, obj in enumerate(box_objects.values()):

            if i%10 == 0:
                print('{} boxes processed....'.format(i))

            if i == 0:
                df_actual = obj.dataframe_from_actual_events()
                df_hr = obj.dataframe_from_hourly_events()

            if i > 0:
                df_actual = df_actual.append(obj.dataframe_from_actual_events())
                df_hr = df_hr.append(obj.dataframe_from_hourly_events())

            i += 1

        # Now save to file
        df_actual.to_csv(output_file_v1, index=False)

        df_hr.to_csv(output_file_v2, index=False)

        print("SUCCESSFULLY SAVED DATASETS.....")
    except Exception as e:
        desired_trace = traceback.format_exc(sys.exc_info())
        print(desired_trace)


def process_raw_sms(sms_observed_file=None, sms_rect_hr_file=None, raw_sms='sms.xml', box_details='Boxes.csv', debug_mode=True):
    """
    Does full processing of the raw data and saves processed files to disk
    :param sms_observed_file: File to be saved  as sms_observed
    :param sms_rect_hr_file: File to be saved  as sms_rect_hr
    :param raw_sms: sms.xml
    :param box_details: Boxes.csv
    :param debug_mode: Whether to debug or not
    :return:
    """
    # ------------ GET BOX METADATA --------------
    box_metadata = box_loc_metadata_as_dict(box_details)

    # ------------ CREATE BOX OBJECTS --------------
    box_objects = create_box_obj_from_events(raw_sms, box_metadata, debug_mode=debug_mode)
    print('SUCCESSFULLY CREATED BOX OBJECTS.........')

    # ------------ SAVE sms_observed.CSV --------------
    print('START SAVING DATA TO FILE...')
    save_datasets(box_objects, sms_observed_file, sms_rect_hr_file)

    # ------------ SAVE BOX-LEVEl-VARIABLES --------------


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


def k_nearest_boxes(box_metadata_file=None, box_id=None, box_lat_lon=None, k=2):
    """
    Selects k-nearest boxes to a refernce box. This includes a box in the same PSU
    :param box_metadata_file:
    :param box_id:
    :param box_lat_lon:
    :param neighbors:
    :return:
    """
    df = pd.read_csv(box_metadata_file, usecols=['LONG', 'LAT', 'ClusterId', 'BoxID'])
    df.rename(columns={'LONG': 'lon', 'LAT': 'lat', 'ClusterId': 'psu', 'BoxID': 'box_id'}, inplace=True)
    bx = df.sort_values(by='psu')

    # remove box id under consideration so that the idea of neighbors makes sense
    bx = bx[bx.box_id != box_id]
    bx.is_copy = False

    bx['dist'] = bx.apply(lambda row: calculate_distance([row['lat'], row['lon']], box_lat_lon), axis=1)

    nearest_n = bx.sort_values(by=['dist'], ascending=True)[:k]

    neighbors = list(nearest_n.box_id.values)

    return neighbors

# TODO this will save BOX level variables
def save_viz_variables(output_file, raw_sms='sms.xml', box_details='Boxes.csv'):
    """
    Saves sms_todaysdate_v1.csv-the first processed file to be outputed.
    :param output_file: Full path of output file
    :param raw_sms: Pretty much sms.xml
    :param box_details: Boxes.csv from where to get box metadata
    :return: Saves to disk
    """
