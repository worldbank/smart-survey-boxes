"""
Contains functions and classes used for data processing
"""

import csv
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from math import radians, cos, sin, atan2, sqrt

import os
import pandas as pd
import re
from collections import Counter
from collections import OrderedDict
from collections import namedtuple
from pytz import timezone as tz

# Outage events
ObservedOutage = namedtuple('ObservedOutage',
                            'box_id start_time end_time duration_hrs in_six_10pm, hrs_in_between_6_10pm')


class Box:
    """
    Holds information about each box and the corresponding data about the box.
    """

    TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
    TIME_FORMAT_TAB1 = '%Y-%m-%d'  # for date collection started

    def __init__(self, box_id=1024, region='Dushnabe', district=None, psu=1, lon=1, lat=2, urban_rural=None,
                 date_collection_started=None):
        self.box_id = box_id
        self.region = region
        self.district = district
        self.psu = psu
        self.lon = lon
        self.lat = lat
        self.date_collection_started = date_collection_started
        self.urban_rural = urban_rural
        self.actual_events = []  # Storing in a set to ensure there are no duplicate events for the same box
        self.hourly_events = None  # Stored in a dictionary for fast access
        self.daily_summaries = OrderedDict()  # Store summariesd variables for each day
        self.num_test_events = None
        self.valid_box_id = True  # put in place to flag boxes whose ID's aren't available in the Boxes.csv
        self.valid_duplicate_events = None  # set based on datetime_sent for events
        self.observed_power_outage_events = None  # keeps observed power_outages
        self.daily_power_outage_duration = None  # Total_hours without power for that day

    def drop_events(self):
        """
        Using the valid attribute of events to keep only the valid events. Also, keeps
        only unique events.
        :return: Number of duplicates
        """
        new_events = []
        for e in self.actual_events:
            if e.valid == 'valid':
                new_events.append(e)

        new_events_without_dups = set(new_events)
        self.actual_events = new_events_without_dups
        self.valid_duplicate_events = len(new_events) - len(new_events_without_dups)

    def drop_events_long_version(self):
        """
        Drops test events and those before data collection started
        :return:
        """
        # print('Number of actual events....{}'.format(len(self.actual_events)))
        new_events = []
        test_events = []
        events_before_start = []

        cnt_test = 0
        for e in self.actual_events:
            if e.before_date_collection or e.event_type_str == 'test':
                if e.before_date_collection:
                    e.box_id = self.box_id
                    e.issue = 'collection before start date'
                    e.date_collection_started = self.date_collection_started
                    events_before_start.append(e)

                if e.event_type_str == 'test':
                    e.box_id = self.box_id
                    e.issue = 'test event'
                    e.date_collection_started = self.date_collection_started
                    test_events.append(e)
                    cnt_test += 1

                continue

            new_events.append(e)

        self.actual_events = new_events
        self.num_test_events = cnt_test
        # print('Number of actual events after dropping events before date of collection....{}'
        #       .format(len(self.actual_events)))

        return test_events, events_before_start

    def add_event(self, event):
        """
        The reason for creating a special set method is to first check
        if this event happened after date-collection started
        :param: event: the event we want to add
        :return: Doesnt return anything
        """
        # Check that the timezones are the same between the box-date-collection started and event date
        self.actual_events.append(event)

    def get_box_metadata(self, required_box_metadata_lst=None):
        """
        A helper function to get get a few selected metadata about a box
        :param required_box_metadata_lst: List of required metadata for this box
        :return: A dict object of box attribute and its value
        """
        all_box_metadata = self.__dict__
        required_box_metadata_dict = {}

        if not required_box_metadata_lst:
            required_box_metadata_lst = ['box_id', 'date_collection_started', 'region', 'district', 'urban_rural',
                                         'psu', 'lon', 'lat', 'valid_duplicate_events']

        for i in required_box_metadata_lst:
            if i in all_box_metadata:
                if i == 'date_collection_started':
                    str_date = self.date_collection_started.strftime(self.TIME_FORMAT_TAB1)
                    required_box_metadata_dict[i] = str_date
                else:
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
                          'event_type_str', 'event_type_num', 'box_state', 'power_state']
        if not box_attr:
            box_attr = ['box_id', 'psu', 'loc', 'date_collection_started']

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

    # TODO: power_on_hours
    def power_on_hours(self):
        """
        For a single day, calculates how many hours power was on
        :return:
        """
        pass

    def generate_hourly_events_based_on_power_state(self, after_event_threshold=13, invalid_threshold=72):
        """
         Creates rectangular dataset at 1-hour by interpolating power-state based on previous event
        :param after_event_threshold: maximum number of hours for carrying foward power-state
        :param invalid_threshold: When to label hours as invalid
        :return:
        """

        # Since its hourly, time interval is hardcoded to 1, otherwise, this could, be done at any time interval
        time_intv = 1

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
            generated_events = self.insert_power_state_for_hours_in_between_events(ev0=first, ev1=second,
                                                                                   time_interval=time_intv,
                                                                                   missing_threshold=after_event_threshold,
                                                                                   perm_missing_threshold=invalid_threshold)
            # update all_events whenever they are available
            if generated_events:
                all_events.update(generated_events)

        # set hourly events property to rect
        self.hourly_events = all_events

    @staticmethod
    def assign_power_state_from_prev_event(prev_event, time_after_prev_ev, is_missing_threshold=13,
                                           is_invalid_threshold=72):
        """
        Assign power state for an inserted event based on the previous event.
        An inserted event is that which is extrapolated.
        Assumption: Events are normal, event consistency is not being verified. Here event consistency
        means that for instance, a pon_mon event shouldn't probably follow pfail event.

        :param prev_event: Previous event
        :param time_after_prev_ev: Number of hours passed after previous event
        :param  is_missing_threshold: Threshold for flagging as missing with reference to previous event
        :param is_invalid_threshold: Threshold for flagging as missing with reference to next event
        :return: Power state (1:power on, 0:power off, -1:missing, 99:invalid)
        """

        if time_after_prev_ev <= is_missing_threshold:
            return prev_event
        elif is_missing_threshold < time_after_prev_ev <= is_invalid_threshold:
            return -1
        elif time_after_prev_ev > is_invalid_threshold:
            return 99

    def insert_power_state_for_hours_in_between_events(self, ev0, ev1, time_interval=1, missing_threshold=13,
                                                       perm_missing_threshold=72):
        """
        Given two observed events, the function interpolates power-states in between these hours.
        The hour in between these two actual events takes the value of the last event as long as the number
        of hours passed is less than given threshold (is_missing_threshold_prev)
        :param time_interval: fixed to 1 hr
        :param ev0: start-event
        :param ev1: end-event
        :param missing_threshold: When to flag blank event as missing based on hours from previous event
        :param perm_missing_threshold: Whether to bale hours as permanently missing or not
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

            # create new event with message body set to 'not_observed'
            event_obj = Event(message='not_observed', datetime_sent=ts, datetime_sent_raw=None)
            event_obj.set_attributes(inserted=True)

            event_obj.power_state = self.assign_power_state_from_prev_event(prev_event=ev0.power_state,
                                                                            time_after_prev_ev=hrs_after_last_event,
                                                                            is_missing_threshold=missing_threshold,
                                                                            is_invalid_threshold=perm_missing_threshold)

            # set attributes for event
            event_obj.data_source = 'insertion'  # source of event data
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
            columns = ['box_id', 'region', 'district', 'urban_rural', 'psu', 'lon', 'lat', 'date_collection_started',
                       'datetime_sent_raw', 'date_sent', 'str_datetime_sent', 'str_datetime_sent_hr', 'event_type_str',
                       'event_type_num', 'power_state', 'ping_event', 'data_source', 'valid', 'valid_duplicate_events']

        # change from dictionary to list
        box_metadata = self.get_box_metadata()

        raw_events = self.actual_events
        actual_events_metadata = []

        df = None
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

    def dataframe_from_hourly_events(self, columns=None, box_metadata=None):
        """
        creates pandas dataframe for convinient usage in other processes
        :return: pandas dataframe
        """
        if not columns:
            columns = ['box_id', 'region', 'district', 'urban_rural', 'psu', 'lon', 'lat', 'datetime_sent_raw',
                       'date_collection_started', 'str_datetime_received', 'str_datetime_sent',
                       'date_sent', 'str_datetime_sent_hr', 'day_sent', 'hour_sent', 'month_sent', 'wk_day_sent',
                       'wk_end', 'event_type_str', 'event_type_num', 'power_state', 'ping_event', 'data_source']
        hr_events_metadata = []

        if not box_metadata:
            box_metadata = self.get_box_metadata()
        df = None
        try:
            for eve in self.hourly_events.values():
                event_attr = eve.get_selected_event_metadata()
                event_attr.update(box_metadata)
                hr_events_metadata.append(event_attr)

            # create dataframe
            df = pd.DataFrame(hr_events_metadata)
            df.sort_values(by='str_datetime_sent_hr', inplace=True)

            # reorder columns
            df = df[columns]
        except Exception as e:
            print('Box ID with issues....{}'.format(box_metadata.get('box_id')))
            print(df.columns)
            print(df.head())
            print(e)

        return df

    def generate_observed_power_out_events(self):
        """
        Generate a dataframe of observed power out events
        :return:
        """
        # keep only pback and pfail events
        power_on_off_events = []
        for event in self.actual_events:
            if event.event_type_str in ['pback', 'pfail']:
                power_on_off_events.append(event)

        # sort events
        sorted_events = sorted(power_on_off_events, key=lambda x: x.datetime_sent)

        # store actual events to this dict object
        outage_events = []

        # Loop through events 2 at a time and create blank events, populate the all_events
        for first, second in zip(sorted_events, sorted_events[1:]):
            first_event = first.event_type_str
            second_event = first.event_type_str

            if first_event + '-' + second_event == 'pfail-pback':
                # outage duration
                duration = (second.datetime_sent - first.datetime_sent).total_seconds() / 3600

                # whether outage is between 6-10pm
                between_6_and_10 = 0
                if first.datetime_sent.hour >= 18 and second.datetime_sent.hour <= 22:
                    between_6_and_10 = 1

                this_event = ObservedOutage(box_id=self.box_id, start_time=first_event.date_sent,
                                            end_time=second_event, duration_hrs=duration,
                                            in_six_10pm=between_6_and_10)
                outage_events.append(this_event)
            else:
                continue

        self.observed_power_outage_events = outage_events

    def dataframe_from_observed_power_outages_event(self):
        """
        Dataframe of outages from observed events
        :return:
        """

        if not self.observed_power_outage_events:
            self.generate_observed_power_out_events()

        df = pd.DataFrame([ev.__dict__ for ev in self.observed_power_outage_events])

        return df

    def generate_daily_power_outage_duration(self, exclude_invalid_hrs=True, exclude_missing_hrs=True):
        """
        Create outage hours per day using hourly events
        :param exclude_invalid_hrs: whether to include invalid hours or not
        :param exclude_missing_hrs: whether to include missing hours or not
        :return: 
        """

        try:
            if not self.hourly_events:
                self.generate_hourly_events_based_on_power_state()

            df_hr = self.dataframe_from_hourly_events()
            if exclude_invalid_hrs:
                df_hr = df_hr[df_hr.power_state != 99]

            if exclude_missing_hrs:
                df_hr = df_hr[df_hr.power_state != -1]

            df_sum = summarize_outages_by_date(df=df_hr, box_id=self.box_id)
            return df_sum
        except Exception:
            pass

    def check_number_of_hr_events(self):
        actual = self.actual_events.sort(key=lambda x: x.datetime_sent)
        start = actual[0]
        print('Start date of events: {}'.format(start.ctime()))
        end = actual[-1]
        print('End date of events: {}'.format(end.ctime()))

        cnt = 0
        ts = start
        while start < end:
            ts = ts + timedelta(hours=1)
            cnt += 1
            yield _

        assert (self.hourly_events.__len__() <= cnt)


class Event:
    """
    An event is a single sms and they are many fields describing an event.
    """
    TAJIK_HOLIDAYS = ['2016-11-06', '2017-01-02', '2017-03-21', '2017-03-22', '2017-03-23',
                      '2017-05-1', '2017-05-9', '2017-06-25', '2017-06-26', '2017-06-27',
                      '2017-09-01', '2017-09-11', '2017-11-06', '2017-03-24']

    # Constants for handling time stamps and conversions
    TIME_ZONE = tz('Asia/Dushanbe')  # using Tajikistan time zone
    OUTPUT_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
    READABLE_DATE_FRMT = '%m/%d/%Y %H:%M:%S'
    TIME_FORMAT_DATE = '%Y-%m-%d'
    REF_DATE = datetime(1960, 1, 1, hour=0, minute=0, second=0, tzinfo=TIME_ZONE)
    TIME_CONSTANT = 315637200000  # The constant 315637200000 is being added as some kind of offset based on stata code

    def __init__(self, datetime_received=None, datetime_sent_raw=None, datetime_sent=None, message=None):
        """
        box-state (-1: unknown, 1: alive, 0: dead)-initialised to -1 for unknown status
        power-state (-1: missing, 1: on, 0: fail, 99: invalid)-initialised to -1 for unknown status
        data_source: A label to indicate source of data (e.g., if based on observed value, we say actual, otherwise it
        could be interpolated (between events based on a rule), imputed (if value was missing, we will impute)

        :param time_stamp: Time stamp of the event
        :param box_id: Box-ID
        :param message: This is extracted from  the body of the message
        """
        self.before_date_collection = False  # for detecting events before start date of data collection
        self.datetime_received = datetime_received  # same as readable_date-not used in any processing steps.
        self.str_datetime_received = None  # same as readable_date-not used in any processing steps.

        # generated from date_sent-THIS IS THE DATE VARIABLE USED IN ALL ANALYSIS
        self.datetime_sent_raw = datetime_sent_raw
        self.datetime_sent = datetime_sent  # the exact datetime generated from raw numeric date above
        self.datetime_sent_hr = None  # same as above, just truncated to hour precision
        self.str_datetime_sent = None  # string representation
        self.str_datetime_sent_hr = None  # string representation
        # the following attributes are for training imputation model
        self.day_sent = None
        self.date_sent = None
        self.hour_sent = None
        self.year_sent = None
        self.month_sent = None
        self.wk_day_sent = None
        self.wk_end = 0  # its not weekend unless otherwise as below
        self.holiday = 0

        # for flagging hours with missing events
        self.hrs_since_event = None  # same as above but here any message counts (relaxed)
        self.hrs_before_event = None  # same as above but here any message counts (relaxed)
        self.is_missing = 0  # indicates whether event data is missing (0-not missing, 1-missing)

        # message body and all variables derived from it
        self.message = message  # message body from sms, for inserted events, message = 'not_observed'
        self.valid = 'valid'  # whether message is valid event: [valid, test, invalid_message, no_id, etc]
        self.event_type_str = None  # generated from message: [pon, pon_mon, pfail, pfail_mon]
        self.event_type_num = None  # numeric representation of above
        self.ping_event = 0  # whether its a ping event: [0,1]
        self.power_state = -1  # set default to unknown: [-1, 99, 1, 0]

        # three sources: observed_event (actual), inserted_event (interpolated between events),
        # imputed_event (only when event is missing)
        self.data_source = None

    def __hash__(self):
        return hash(self.datetime_sent)

    def __eq__(self, other):
        if isinstance(other, Event):
            return self.datetime_sent == other.datetime_sent
        return NotImplemented

    def set_datetime_sent_from_numeric_date_sent(self):
        """
        Set date time
        :return:
        """
        if self.datetime_sent_raw:
            if int(self.datetime_sent_raw) == 0:
                self.datetime_sent = self.datetime_received
                return
            self.datetime_sent = self.convert_datetime_sent(self.datetime_sent_raw)

    def set_attributes(self, inserted=False, date_collection=None, device_id=None, valid_box_id=None):
        """
        set all attributes
        :param inserted:
        :param date_collection:
        :param device_id:
        :param valid_box_id:
        :return:
        """
        # set datetime_sent
        self.set_datetime_received(date_str=self.datetime_received)
        self.set_datetime_sent_from_numeric_date_sent()
        self.datetime_sent_hr = self.datetime_sent.replace(minute=0, second=0)

        # --------DATETIME and associated variables which is used for all time-stamp purposes-------
        if isinstance(self.datetime_received, datetime):
            self.str_datetime_received = self.datetime_received.strftime(self.OUTPUT_TIME_FORMAT)
        else:
            self.str_datetime_received = self.datetime_received

        self.str_datetime_sent = self.datetime_sent.strftime(self.OUTPUT_TIME_FORMAT)
        self.str_datetime_sent_hr = self.datetime_sent_hr.strftime(self.OUTPUT_TIME_FORMAT)
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

        # --------TYPE OF EVENT VARIABLES--------
        event_type = {'test': 0, 'pback': 1, 'pfail': 2, 'pon_mon': 3, 'pfail_mon': 4, 'missing': -1}
        if not inserted:
            self.event_type_str = self.detect_event_type_from_message(message_body=self.message)  # category of event

        self.event_type_num = event_type.get(self.event_type_str)  # same as abobe but use numeric values

        # POWER STATE: unknown (-1), power on (1), power-off (0)
        power_state_dict = {'pback': 1, 'pfail': 0, 'pon_mon': 1, 'pfail_mon': 0}
        self.power_state = -1  # set default to unknown
        self.power_state = power_state_dict.get(self.event_type_str)

        # whether event happened before date collection started
        try:
            if self.datetime_sent <= date_collection:
                # print('Event date {} is before date of collection: {} for box {}....'
                #       .format(event_date_str, collection_date_str, self.box_id))
                self.before_date_collection = True
        except TypeError:
            self.before_date_collection = None

        # set validity of this event
        valid_str = ''
        if self.event_type_str == 'test':
            valid_str += 'test,'

        if device_id == -99:
            valid_str += 'no-device-id,'

        if self.before_date_collection:
            if self.before_date_collection:
                valid_str += 'before-date-of-collection,'

        if self.event_type_str == 'invalid_message':
            valid_str += 'invalid-message,'

        if not valid_box_id:
            valid_str += 'no-box-id-in-box-file'

        if valid_str:
            self.valid = valid_str

    def set_datetime_received(self, date_str=None):
        """
        Fix readable date based on the fact that Russioan characters are being read like this:
        07 ???. 2016 ?. 18:29:39
        :return: Date String
        """
        try:
            if len(date_str) == 24:
                new_date_str = '12/' + date_str[:2] + '/2016' + ' ' + date_str[-8:]
                self.datetime_received = new_date_str
            elif len(date_str) == 23:
                new_date_str = '12/' + date_str[:2] + '/2016' + ' ' + '0' + date_str[-7:]
                self.datetime_received = new_date_str

            # convert to datetime object
            date = datetime.strptime(self.datetime_received, self.READABLE_DATE_FRMT).replace(tzinfo=self.TIME_ZONE)
            self.datetime_received = date
        except Exception:
            pass

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
            date_str = date_value.strftime(self.TIME_FORMAT_DATE)

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

    def detect_event_type_from_message(self, message_body=None):
        """
        Looks at the message and classifies it as follows:
        test-Messages with test in them
        pon_mon-power on monitoring
        pfail_mon-power off monitoring
        pfail-power fail
        :return: a string for the type of event
        """
        msg = message_body
        msg_2 = self.split_message()[1]

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
            event_type = 'invalid_msg'

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
        if all_event_metadata['date_sent']:
            all_event_metadata['date_sent'] = self.datetime_sent.strftime(self.TIME_FORMAT_DATE)

        required_event_metadata_dict = {}

        if not required_event_metadata:
            required_event_metadata = ['datetime_sent_raw', 'str_datetime_received', 'str_datetime_sent', 'datetime_sent',
                                       'str_datetime_sent_hr', 'datetime_sent_hr', 'day_sent', 'date_sent',
                                       'hour_sent', 'month_sent', 'wk_day_sent', 'wk_end', 'holiday', 'hrs_since_event',
                                       'hrs_before_event', 'event_type_str', 'event_type_num', 'power_state',
                                       'ping_event', 'valid', 'data_source']

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


def power_outage_within_6_10pm(x=None):
    if 18 <= x <= 22:
        return 1

    return 0


def date_range(date1, date2):
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + timedelta(n)


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


def extract_box_id(message):
    """
    Extracts Box-ID from the message body
    :return: BOX-ID
    """
    digits_in_msg = re.findall(r'\d+', message)
    if len(digits_in_msg) < 1:
        # print('It seems there is no box_id for this message--- {}'.format(message))
        return -99
    else:
        return digits_in_msg[0]


def convert_to_aware_datetime_object(date_str, time_format='%m/%d/%Y %H:%M:%S'):
    """
    A helper function to convert from string to
    :return: A time zone aware time object
    """
    time_zone = tz('Asia/Dushanbe')

    date_obj = datetime.strptime(date_str, time_format)
    date_obj = date_obj.replace(tzinfo=time_zone)

    return date_obj


def generate_event_transitions(df: pd.DataFrame):
    """
    Generates event transitions
    :return:
    """
    events = list(df.event_type_str)
    transition_matrix = pd.DataFrame(index=list(set(events)), columns=list(set(events)))

    for (x, y), c in Counter(zip(events, events[1:])).items():
        transition_matrix.loc[x][y] = c

    return transition_matrix


def add_elapsed_time_column(df: pd.DataFrame, units='hr', time_col='str_datetime_sent'):
    """
    Returns dataframe with extra column of elapased time in hours using 
    pandas functionality
    :param df: 
    :return: 
    """
    # Set time_col as index
    df2 = df.set_index(df[time_col])
    df2['time_elapsed'] = (df2[time_col] - df2[time_col].shift()).fillna(0)

    if units == 'hr':
        df2['time_elapsed_hrs'] = df2['time_elapsed'].apply(lambda x: int(x.total_seconds() / 3600))

    df2 = df2.set_index([[i for i in range(df2.shape[0])]])

    return df2


def add_elapsed_time_to_dataframe(df: pd.DataFrame, units='hr', time_col='str_datetime_sent'):
    """
    We could probably just order the dataframe by box and time BUT nah, just
    not sure what happens at boundaries with this approch, so doing this
    for each box separately.
    :param df:
    :param units:
    :param time_col:
    :return:
    """
    out_df = None
    for i, bx in enumerate(list(df.box_id.unique())):
        bx = df[df.box_id == bx]
        df_bx = add_elapsed_time_column(bx)

        if i == 0:
            out_df = df_bx
        else:
            out_df = out_df.append(df_bx)

    out_df = out_df.set_index([[i for i in range(out_df.shape[0])]])

    return out_df


def convert_date_to_string(date_obj):
    time_format = '%Y-%m-%d %H:%M:%S'
    date_1_str = date_obj.strftime(time_format)

    return date_1_str


def xml_to_list_of_dicts(xml_file=None, msgs=True):
    """
    Return a list contaning dict objects from xml elements
    :param xml_file:
    :param msgs:
    :return:
    """
    print('=' * 70)
    print(' Converting sms events in xml file to python dict .....')
    print('=' * 70)

    # list to hold the data as dict objects  retrieved from xml
    lst = []
    try:
        et = ET.parse(xml_file).getroot()
        it = et.getiterator()
        while True:
            try:
                item = it.__next__().items()
                if len(item) > 10:
                    lst.append(item)
                else:
                    print('The record below was excluded because it doesnt look like an event...')
                    print(item)
            except StopIteration:
                break  # Iterator exhausted: stop the loop
        header = list(dict(lst[0]).keys())
        if msgs:
            num_events = et.__len__()
            print("About this xml file==> Date last modified: %s" % time.ctime(
                os.path.getmtime(xml_file)))
            print('Number of events in this sms.xml: {:,}'.format(num_events))
            print("Event attributes are shown below...")
            print(header[:8])
            print(header[8:])
    except Exception as e:
        print(e)
        print('Reading xml failed....')

    return lst


def convert_xml_to_csv(data_processor_obj=None, ts=None, verification_msgs=True):
    """
     Return csv from xml.
    :param data_processor_obj:
    :param ts:
    :param verification_msgs:
    :return:
    """
    print('=' * 50)
    print('XML==>CSV conversion started.....')
    print('=' * 50 + '\n')

    # convert xml to list of dict objects
    xml_file = data_processor_obj.xml_file
    lst = xml_to_list_of_dicts(xml_file=xml_file, msgs=verification_msgs)
    lst_dict = [dict(i) for i in lst]
    header = list(lst_dict[0].keys())

    # csv file to write to
    csv_file = os.path.join(data_processor_obj.raw_data_dir, 'sms_' + ts.strftime('%m-%d-%Y') + '.csv')

    try:
        with open(csv_file, 'w', encoding='UTF-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            for item in lst:
                data = dict(item)
                writer.writerow(data)
    except Exception as e:
        print('Failed to write to file %s' % e)

    print()
    print('XML==>CSV conversion finished.')


def log_processing_message(msgs_with_no_id=None, boxes_with_no_events=None, test_events=None, start_date_events=None,
                           outfile=None, out_csv_file=None, box_file=None, message_type_prob=None):
    """
    For metdata purposes, indicated how many events were omitted
    :param msgs_with_no_id:
    :param box_with_few_events:
    :param outfile: File containing the logs
    :return:
    """
    # combine events
    if message_type_prob:
        all_events = msgs_with_no_id + test_events + start_date_events + message_type_prob
    else:
        all_events = msgs_with_no_id + test_events + start_date_events

    all_events_data = []

    for e in all_events:
        ev = {}
        if type(e) == Event:
            ev['box_id'] = e.box_id
            ev['readable_date'] = e.datetime_received
            ev['date_sent'] = e.datetime_sent
            ev['date_collection_started'] = e.date_collection_started
            ev['issue'] = e.issue
            ev['message'] = e.message
            ev['phonenumber'] = None
        else:
            ev['box_id'] = e.get('box_id', -1)
            date_str = e['datetime_sent']

            # still one event object inorder to use its method
            ev_tmp = test_events[0]
            try:
                date_obj = ev_tmp.convert_datetime_sent(value=date_str)
            except Exception as ex:
                try:
                    date_obj = datetime.strptime(e['readable_date'], '%m/%d/%Y %H:%M:%S')
                except Exception as ex:
                    date_obj = None

            ev['readable_date'] = e['readable_date']
            ev['date_sent'] = date_obj
            ev['date_collection_started'] = None
            ev['issue'] = e.get('issue')
            ev['message'] = e.get('message')
            ev['phonenumber'] = e.get('phonenumber')

        all_events_data.append(ev)

    # create df
    df = pd.DataFrame(all_events_data)
    df.to_csv(out_csv_file, index=False)
    df_id = df[df.box_id != -1]
    uniq_vals = pd.unique(df_id[['box_id', 'date_sent']].values.ravel(1))

    wr = open(outfile, 'w')

    # write header for omitted events
    wr.write('-' * 80 + '\n')
    wr.write("About this box file==> Date last modified: {}".format(time.ctime(os.stat(box_file).st_mtime)))
    wr.write('\n')
    wr.write('-' * 80 + '\n')
    wr.write('\n')

    # write header for omitted boxes
    wr.write('=' * 80 + '\n')
    wr.write('BOXES DROPPED BECAUSE THEY HAVE NO EVENTS AFTER DROPPING INVALID EVENTS :{}'.format(boxes_with_no_events))
    wr.write('\n')
    wr.write('=' * 80 + '\n')
    wr.write('\n')

    # write header for omitted events
    wr.write('=' * 80 + '\n')
    wr.write('{:,} EVENTS DROPPED DUE TO MISSING DEVICE ID IN MESSAGE BODY'.format(len(msgs_with_no_id)))
    wr.write('\n')
    wr.write('=' * 80 + '\n')

    # write header for events dropped because they are test events
    wr.write('\n')
    wr.write('=' * 80 + '\n')
    wr.write('{:,} TEST EVENTS WERE DROPPED'.format(len(test_events)) + '\n')
    wr.write('=' * 80 + '\n')
    wr.write('\n')

    # write header for events dropped because they are test events
    wr.write('\n')
    wr.write('=' * 80 + '\n')
    wr.write('{:,} EVENTS WHERE MESSAGE TYPE CANT BE DETERMINED'.format(len(test_events)) + '\n')
    wr.write('=' * 80 + '\n')
    wr.write('\n')

    # write header for events dropped because they occur after date collection started
    wr.write('\n')
    wr.write('=' * 80 + '\n')
    wr.write('{:,} EVENTS WHICH OCCUR AFTER DATE START COLLECTION WERE DROPPED'.format(len(start_date_events)) + '\n')
    wr.write('=' * 80 + '\n')
    wr.write('\n')

    wr.write('\n')
    wr.write('=' * 80 + '\n')
    wr.write('** FOR EVENTS WITH BOX-ID AVAILABLE, {:,} UNIQUE EVENTS WERE DROPPED DUE TO DATE ISSUE OR TEST EVENT '
             '**'.format(uniq_vals.shape[0]) + '\n')
    wr.write('=' * 80 + '\n')
    wr.write('Please look at the *dropped-events.csv* for details of these events')
    wr.write('\n')

    wr.close()


def generate_event_objects(box_raw_events=None, box_obj=None):
    """
    For a single box, a list of raw events is passed and it creates events with the required attributes.
    :param box_raw_events:
    :param box_obj:
    :return: All events for this box and list of events where a message type cant be determined.
    """
    for item in box_raw_events:
        # create an event object with core attributes
        event = Event(datetime_received=item.get('readable_date'), datetime_sent_raw=item.get('date_sent'),
                      message=item.get('body'))

        # set all important attributes
        event.set_attributes(inserted=False, device_id=box_obj.box_id, date_collection=box_obj.date_collection_started,
                             valid_box_id=box_obj.valid_box_id)

        # indicate data source
        event.data_source = 'observed_event'

        # whether its ping event or not
        if event.event_type_str in ['pfail_mon', 'pon_mon']:
            event.ping_event = 1  # helper variable for counting events

        box_obj.add_event(event)

    return box_obj


def event_dict_from_xml(xml_path=None):
    """
    Create a dict object with events and keyed by box-id
    :param xml_path: The sms.xml file
    :return: A dictionary of events indedxed by box id
    """
    # grab all events from xml
    lst = xml_to_list_of_dicts(xml_file=xml_path, msgs=True)
    lst_dict = [dict(i) for i in lst]

    # add box-ids to this dict
    lst_dict_box = {}

    for ii in lst_dict:
        box_id = extract_box_id(ii.get('body'))

        if int(box_id) in lst_dict_box:
            lst_dict_box[int(box_id)].append(ii)
        else:
            lst_dict_box[int(box_id)] = [ii]

    return lst_dict_box


def summarize_outages_by_date(df=None, box_id=None):
    """
    Summarises outages per day for a single box
    :return: df with by date summary
    """
    
    if 'power_off' not in list(df.columns):
        df.loc[:, 'power_off'] = df['power_state'].apply(lambda x: 1 if x == 0 else 0)
        df = df[df.power_off == 1]  # keep only power_off events
        df.is_copy = False
        df.loc[:, 'power_off_10pm'] = df['hour_sent'].apply(lambda x: power_outage_within_6_10pm(x))

    # summarise by date
    df_grp_all = df.groupby(['date_sent', 'power_off'])['power_off'].sum()
    df_grp_all = pd.DataFrame(df_grp_all)
    df_grp_all = df_grp_all.reset_index(level=1, drop=True).reset_index()
    df_grp_all.rename(columns={'power_off': 'hrs_power_off'}, inplace=True)

    df_grp_10 = df.groupby(['date_sent', 'power_off_10pm'])['power_off_10pm'].sum()
    df_grp_10 = pd.DataFrame(df_grp_10)
    df_grp_10 = df_grp_10.reset_index(level=1, drop=True).reset_index()
    df_grp_10.rename(columns={'power_off_10pm': 'hrs_power_off_10pm'}, inplace=True)

    df_out = pd.merge(left=df_grp_all, right=df_grp_10, on='date_sent')
    df_out['box_id'] = box_id

    return df_out


def summarise_outage_counts_from_stata_powerout_file(powerout_file=None):
    """
    Since his file has a differen structure, I use his function to summarise it
    :return: DataFrame
    """

    df = pd.read_csv(powerout_file)
    df = df[df.POWERout == 1]  # keep only power_off events
    df['date_sent'] = df['date_powerfailure'].apply(lambda x: datetime.strftime(datetime.strptime(x, '%d%b%Y'), '%Y-%m-%d'))
    df.rename(columns={'BoxID': 'box_id', 'POWERout': 'power_off', 'dhms': 'datetime_sent',
                       'date_powerfailure_hour': 'hour_sent'}, inplace=True)
    df = df[['box_id', 'power_off', 'date_sent', 'hour_sent', 'datetime_sent']]
    df['power_off_10pm'] = df['hour_sent'].apply(lambda x: power_outage_within_6_10pm(x))
    df_all = None

    i = 0
    for bx_id in list(df.box_id.unique()):
        bx_df = df[df.box_id == bx_id]
        bx_df.is_copy = False
        if i == 0:
            df_all = summarize_outages_by_date(df=bx_df, box_id=bx_id)

        if i > 0:
            df_tmp = summarize_outages_by_date(df=bx_df, box_id=bx_id)
            df_all = df_all.append(df_tmp)

        i += 1

    return df_all


def calculate_daily_average_power_out(df=None, var=None):
    """
    Average durations for each day across all boxes
    :return:
    """
    df_grp = df.groupby(['date_sent'])[var].mean()
    df_grp = pd.DataFrame(df_grp).reset_index()
    df_grp.sort_values(by='date_sent', inplace=True)
    df_grp.insert(0, 'day', range(1, len(df_grp) + 1))
    df_grp.rename(columns={var: 'avg_' + var}, inplace=True)
    
    return df_grp


def normalise_power_out_events(df: pd.DataFrame=None):
    df2 = df.__deepcopy__()

    if df2.iloc[0].event_type_str == 'pback':
        df2 = df2[1:]

    lst_events = [df2.iloc[i] for i in range(len(df2))]

    grped_lst_events = list(group(lst=lst_events, n=2))

    invalid_power_out_events = 0
    new_event_list = []

    for i in grped_lst_events:
        event_1 = i[0]
        event_2 = i[1]
        event_str = event_1.event_type_str + '-' + event_2.event_type_str
        if event_str == 'pfail-pback':
            new_event_list = new_event_list + [event_1, event_2]
        else:
            invalid_power_out_events += 1
            continue

    return invalid_power_out_events, pd.DataFrame(new_event_list)


def group(lst=None, n=None):
    """group([0,3,4,10,2,3], 2) => [(0,3), (4,10), (2,3)]

    Group a list into consecutive n-tuples. Incomplete tuples are
    discarded e.g.

    >>> group(range(10), 3)
    [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    """
    return zip(*[lst[i::n] for i in range(n)])


def add_time_interval(df=None, datetime_col=None):
    """
    Given a sorted dataframe, adds a time interval column
    """
    df2 = df.__deepcopy__()

    df2['tdiff'] = (df2[datetime_col] - df2[datetime_col].shift()).fillna(0)

    df2['duration_minutes'] = df2['tdiff'].apply(lambda r: r.total_seconds() / 60)

    df2.drop('tdiff', axis=1, inplace=True)

    return df2


def powerout_duration_with_invalid(in_df=None, date_col=None):
    """
    :param df: Dataframe of pfail and pback events sorted by time
    """
    # Just ensure that first event isnt pback: thats the only check we do
    if in_df.iloc[0].event_type_str == 'pback':
        in_df = in_df[1:]

    # add time-interval
    df_with_intervals = add_time_interval(df=in_df, datetime_col=date_col)

    return df_with_intervals


def powerout_duration_with_valid_powerout_events(in_df=None, date_col=None):
    """
    My solution is to take omit all invalid powerout events.
    Invalid powerout events are those not in sequence pfail-pback
    :return:  Dataframe with a set of events which form valid pfail-pback event sequence and their duration
    """

    # Remove invalid powerout events
    invalid_power_out_events, df_valid = normalise_power_out_events(df=in_df)

    # add time-interval
    df_with_intervals = add_time_interval(df=df_valid, datetime_col=date_col)

    return df_with_intervals


def powerout_duration(df=None, exclude_invalid=False, time_col='str_datetime_sent'):
    """
    Create powerout.csv for tableau
    """
    df = df[df['event_type_str'].isin(['pfail', 'pback'])]
    sorted_df = df.sort_values(by=[time_col])

    if exclude_invalid:
        df_with_intervals = powerout_duration_with_valid_powerout_events(in_df=sorted_df, date_col=time_col)
    else:
        df_with_intervals = powerout_duration_with_invalid(in_df=sorted_df, date_col=time_col)

    return df_with_intervals




