import pytz
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
import re
from IPython.display import Image
import itertools
from random import randint, sample


drive = 'G'
if drive == 'G':
    data_dir = '../01.data/'
    output_dir = '../05.outputs/'
else:
    data_dir = '..\\01.data\\'


# Just interpolating events to make sure the matrix isnt sparse
def interpolate_pwr_state(df, t0, t1, e0, max_intv):
    # Deduce power state from message
    # Lets check duration of this period
    diff_hrs = ((t1 - t0).total_seconds()) / 3600

    # print (t1.ctime(),'--',t1.ctime())
    # print ('Duration between events is %s hours'%diff_hrs)
    mask = (df['datetime_rcvd_hr'] >= t0) & (df['datetime_rcvd_hr'] <= t1)
    ping_msgs = df.loc[mask].ping_msg.sum()
    # print (ping_msgs)

    # if duration is <=24 hours and number of monitoring messages is >= 2 interpolate pwr_state

    if diff_hrs <= max_intv:
        # scenario one poff
        if ping_msgs >= diff_hrs / 12:
            if e0 == 'pfail' or e0 == 'pfail_mon':
                return 0
            elif e0 == 'pon_mon' or e0 == 'pback':
                return 1
        else:
            return -1
    else:
        return -1


def interpolate_events(df, power_state, max_hrs):
    df = df.sort_values(by=['datetime_rcvd_hr'])

    events = df[['datetime_rcvd_hr', 'msg']]

    # extract only events-hopefully they are still in sorted order
    events = events.dropna(axis=0, how='any')

    # index of events
    idx = events.index

    # a counter
    i = 0

    # Loop through all events (2 at a time) and extract corresponding rows in df
    while i < len(idx) - 1:

        t0 = events.ix[idx[i], 'datetime_rcvd_hr']

        t1 = events.ix[idx[i + 1], 'datetime_rcvd_hr']

        diff_hrs = ((t1 - t0).total_seconds()) / 3600

        # Event at start hour
        ev0 = events.ix[idx[i], 'msg']

        # Event at end hour
        ev1 = events.ix[idx[i + 1], 'msg']

        # Rows in df matching events in this interval
        mask = (df['datetime_rcvd_hr'] > t0) & (df['datetime_rcvd_hr'] < t1)
        # print (len(mask[mask==True]))


        # Set all events in the range to ev0
        df.ix[mask, 'msg'] = ev0

        # Set source as 'interpolated'
        df.ix[mask, 'msg_src'] = 'interpolated'

        # Also interpolate power state
        if power_state:
            # increment counter
            pwr = interpolate_pwr_state(df, t0, t1, ev0, max_hrs)
            # set power-state to return value
            # Set all events in the range to ev0
            df.ix[mask, 'pwr_state'] = pwr

            # Set source as 'interpolated'
            df.ix[mask, 'pwr_state_src'] = 'interpolated'

        i += 1

    return df

e = pd.read_csv(data_dir + 'smsV2.csv',parse_dates=['datetime_rcvd'])

e = e.sort_values (by=['device_id','datetime_rcvd'])

#Lets add a time stamp which truncates to hour
e['datetime_rcvd_hr'] = e['datetime_rcvd'].apply (lambda x: datetime (x.year, x.month, x.day, x.hour))


start = e.datetime_rcvd_hr.min()
end = e.datetime_rcvd_hr.max()
delta = timedelta(hours = 1)
print ('Events run from :  %s  to :   %s ' %(start.ctime(),end.ctime()))

dates = [start]
step = timedelta(hours=1)

lst_boxes = list (e.device_id.unique())

while start <= end:
    start += step
    dates.append(start)

tuples = list(itertools.product(*[lst_boxes,dates]))

print ('Number of date-hour-box pair is %s' %len(dates))

idx = pd.MultiIndex.from_tuples(tuples, names=['device_id', 'datetime_rcvd_hr'])

#Ideal hourly dataframe would look like this: eb should mean events blank
eb = pd.DataFrame(index=idx)

eb = eb.sort_index()

eb = eb.reset_index()

ev = pd.merge(left=eb,right=e, on=['device_id','datetime_rcvd_hr'],how='left')

ev2 = pd.DataFrame(columns=ev.columns)

for bx in list(ev.device_id.unique()):
    df_bx = ev[ev.device_id == bx]

    last_event = df_bx.datetime_rcvd.max()
    ts_upper = datetime(last_event.year, last_event.month, last_event.day, last_event.hour)

    first_event = df_bx.datetime_rcvd.min()
    ts_lower = datetime(first_event.year, first_event.month, first_event.day, first_event.hour)

    mask = (df_bx['datetime_rcvd_hr'] >= ts_lower) & (df_bx['datetime_rcvd_hr'] <= ts_upper)

    df_bx = df_bx.loc[mask]

    ev2 = ev2.append(df_bx)

# Replace ev with ev2
ev = ev2


ev['ping_msg'] = 0

ev.ix[ev.msg=='pfail_mon',['ping_msg']] =  1

ev.ix[ev.msg=='pon_mon',['ping_msg']] =  1

ev = ev.sort_values (by = ['datetime_rcvd_hr'])

# Initialise box_state to 0
ev['box_state'] = 0

ev = ev.sort_values(by=['device_id', 'datetime_rcvd_hr'])

start = ev.datetime_rcvd_hr.min()

end_all = ev.datetime_rcvd_hr.max()

step = timedelta(hours=24)

while start <= end_all:
    end = start + step

    mask = (ev['datetime_rcvd_hr'] >= start) & (ev['datetime_rcvd_hr'] <= end)

    check_sum = ev.loc[mask].ping_msg.sum()

    if check_sum >= 2:
        ev.ix[mask, 'box_state'] = 1
    else:
        ev.ix[mask, 'box_state'] = 0

    start += step

ev['pwr_state'] = np.nan

#To indicate source of pwr_state information
ev['pwr_state_src'] = ''

#When message is pon_monitoring or power back, set power status to 1
on_mask = ev['msg'].isin(['pback','pon_mon'])

ev.ix[on_mask,'pwr_state'] = 1

ev.ix[on_mask,'pwr_state_src'] = 'actual'

off_mask = ev['msg'].isin(['pfail','pfail_mon'])

ev.ix[off_mask,'pwr_state'] = 0

ev.ix[off_mask,'pwr_state_src'] = 'actual'

# Label actual events as 'actual
# and rest of the events as interpolated
actual_msgs = ev.loc[ev.msg.notnull()].index

ev.ix[actual_msgs, 'msg_src'] = 'actual'

boxes = list(ev.device_id.unique())

# Load new data into this empty dataframe
evi = pd.DataFrame(columns=ev.columns)

for i, b in enumerate(boxes):
    if i % 25 == 0:
        print('Working on box %s of %s' % (i, len(boxes)))

    df_bx = ev[ev.device_id == b]

    evi = evi.append(interpolate_events(df_bx, True, 48))


bx = pd.read_csv(data_dir + 'Boxes.csv',parse_dates=['DateCollectionStart'])

bx = bx[['ClusterId','DateCollectionStart','LONG','LAT', 'BoxID']]

bx.rename(columns={'ClusterId': 'psu','LONG':'lon','LAT':'lat','BoxID':'device_id'},inplace=True)
