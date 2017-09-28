

```python
#Import packages
import random
import os, sys
import pytz
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
import re
from IPython.display import Image
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from pypower import prediction_models as pred
from pypower import data_utils as ut
from pypower import model_selection_custom as cust_mod
from sklearn.metrics import classification_report
from sklearn.ensemble import ExtraTreesClassifier as etc
```


```python
data_dir = "/Users/dmatekenya/Google Drive/World-Bank/electricity_monitoring/01.data/processed_sms/"
```


```python
# Get the data: sms_rect_hr.csv'
cols_to_use = ['box_id', 'psu', 'lon', 'lat', 'str_datetime_sent_hr', 'day_sent', 'hour_sent', 'month_sent',
                   'wk_day_sent', 'wk_end', 'event_type_num', 'event_type_str', 'power_state', 'data_source']
sms2 = pd.read_csv(data_dir + 'sms_rect_hr.csv', usecols=cols_to_use, parse_dates=['str_datetime_sent_hr'])
sms2.rename(columns={'str_datetime_sent_hr': 'datetime_sent_hr'}, inplace=True)

# Remove missing values
sms2 = sms2[sms2.power_state != -1]
```

**EVALUATING NEAREST NEIGHBOR MODEL**
=======================
==================================================================

1. * Pick a test box*
----------------------


```python
# Pick a test box
test_box_id = 1005
data_test_bx = sms2[sms2.box_id == test_box_id]  # select data for this box only

# List of dates, to help with picking random dates for testing
event_dates = list(data_test_bx.datetime_sent_hr)
```

2. * Decide how many tests to do *
--------------------------------------
-Cant do complete leave one out due to time


```python
prop_test = 0.2
max_test_cases = 250
min_test_cases = 50
num_tests = int(data_test_bx.shape[0] * prop_test)
if num_tests < min_test_cases:
        print ('Lets choose another box!!!')

if num_tests > max_test_cases:
    num_tests = max_test_cases

print('We will test {} events out of {} events for this box'.format(num_tests, data_test_bx.shape[0]))
```

    We will test 250 events out of 5530 events for this box


3. * Randomly pick test dates *
--------------------------------


```python
test_dates = random.sample(event_dates, num_tests)

# Generated test events from the dates
test_df = data_test_bx[data_test_bx['datetime_sent_hr'].isin(test_dates)]
```

4. * Remove all test events from training data*
---------------------------------------------


```python
# sms2 minus test box
sms2_without_test_box = sms2[sms2.box_id != test_box_id]

# Now, we only need to remove the exact test events
to_keep = list(set(event_dates) - set(test_dates))

# Within the test box, we can keep the non-test events
test_box_to_keep = data_test_bx[data_test_bx['datetime_sent_hr'].isin(to_keep)]

# Finally, our training dataset is sms2 minus test events in test box
train_df = sms2_without_test_box.append(test_box_to_keep)
```

5. * Just to be sure, check that test events arent in the train dataset*
-------------------------------------------------------------------------
Note: Only for the test-box


```python
# To make it quicker, we only check events in the test box-makes sense 
train_df_test_box = train_df[train_df.box_id == test_box_id]

print('Checking if the training dataset has any test events...')
leaked = 0
for date in test_dates:
    if date in list(train_df_test_box.datetime_sent_hr):
        print ('WAIT A MINUTE, HOW COME TEST EVENTS ARE STILL IN TRAINIGNG DATA')
        leaked += 1
        
if leaked == 0:
    print('Done, found no text events in training data')
```

    Checking if the training dataset has any test events...
    Done, found no text events in training data


6. * Create a nearest neighbor model*
--------------------------------------
This model has the following parameters:
 
 -*target*: what to predict(either power_state or event_type)
 
 -*neighbors*-Number of boxes (based on location to include. e.g., 0 neighbors include 2 boxes)
 
 -*time-window*: moving winodw to search from
 
 -*direction*: whethere to pool foward looking or backward looking events only (centred on test date)
 
 -*how*: How to make prediction, default is frequent
 
 -*train_data*: the training data


```python
# ----------------CREATE MODEL OBJECT-----------------------------
predictor_params = {'neighbors': 1, 'time-window': 7, 'direction': 'both', 'how': 'frequent',
                    'target': 'power_state'}
# model object
clf = pred.ImputationNearestNeighbor(data=train_df, target=predictor_params['target'],
                                         neighbors=predictor_params['neighbors'],
                                         how=predictor_params['how'],
                                         time_window=predictor_params['time-window'],
                                         direction=predictor_params['direction']
                                         )
# location details for boxes
box_file = "/Users/dmatekenya/Google Drive/World-Bank/electricity_monitoring/01.data/Boxes.csv"
clf.generate_box_metadata(box_file=box_file)

# Get lat-lon for the test box
box_lat_lon = [data_test_bx[data_test_bx.box_id == test_box_id].lat.values[0],
                   data_test_bx[data_test_bx.box_id == test_box_id].lon.values[0]]
```

7. * We are now ready to make predictions using the model *
---------------------------------------------------------

7.1 Select neighbors
-------------------------


```python
# search neighbors from all boxes except test box
bx = clf.BOX_METADATA[clf.BOX_METADATA.box_id != test_box_id]
bx.is_copy = False

# compute distance between target (test box) and rest of the boxes
target_loc = box_lat_lon
bx['dist'] = bx.apply(lambda row: ut.calculate_distance([row['lat'], row['lon']],target_loc), axis=1)

# Get top-n nearest neighbors
nearest_n = bx.sort_values(by=['dist'], ascending=True)[:clf.neighbors]
print('The distances...')
print()
print(bx.sort_values(by=['dist'], ascending=True).head())

neighbors = list(nearest_n.box_id.values)
print()

neighbors.append(test_box_id) # since we also want to learn from the same box
print('Given number of neighbors = {}, we have these neighbors: {}'.format(clf.neighbors, neighbors))

print('COMPARE TO OUT TEST BOX DETAILS')
test_box_psu = train_df[train_df.box_id==test_box_id].iloc[0].psu
print('Test box id==> {}, test-box-psu==> {}, test-box-lat/lon==> {}'.format(test_box_id, test_box_psu, box_lat_lon))
```

    The distances...
    
         box_id  psu        lon        lat      dist
    267    1271   94  69.385761  37.529461  0.000000
    253    1257   92  69.395822  37.491284  4.341864
    209    1212   92  69.395822  37.491284  4.341864
    107    1108   93  69.429172  37.575373  6.387610
    135    1136   93  69.429172  37.575373  6.387610
    
    Given number of neighbors = 1, we have these neighbors: [1271, 1005]
    COMPARE TO OUT TEST BOX DETAILS
    Test box id==> 1005, test-box-psu==> 94, test-box-lat/lon==> [37.529461095977027, 69.385761185057504]


7.2 Generate training data
---------------------------
This discards all the excepet that in [test_date-window_length, test_date+window_length]. 
Also, we only keep data for the neighbors.


```python
# Lets pick a test date-first date
prediction_date = test_df.iloc[0].datetime_sent_hr
actual_0 = test_df.iloc[0].power_state
print(test_df.iloc[0])

training_data= clf.generate_train_data(target_date=prediction_date, raw_data=train_df, boxes=neighbors)
```

    box_id                             1005
    psu                                  94
    lon                             69.3858
    lat                             37.5295
    datetime_sent_hr    2017-01-13 21:00:00
    day_sent                             13
    hour_sent                            21
    month_sent                            1
    wk_day_sent                           4
    wk_end                                0
    event_type_str                  pon_mon
    event_type_num                        3
    power_state                           1
    data_source                   insertion
    Name: 1541623, dtype: object



```python
print('---------------------------------------------------')
print ('Checking that training data only has the 2 boxes')
print('---------------------------------------------------')
print(training_data.box_id.value_counts())

print()
print('-----------------------------------------------------------------------------')
print ('Checking that training data is within the time window centred on test date')
print('-----------------------------------------------------------------------------')
print()
print ('#### Test date ==> {}, window-length ==> {} days #######'.format(prediction_date, predictor_params['time-window']))
print(training_data.datetime_sent_hr.describe())
```

    ---------------------------------------------------
    Checking that training data only has the 2 boxes
    ---------------------------------------------------
    1005    209
    1271     28
    Name: box_id, dtype: int64
    
    -----------------------------------------------------------------------------
    Checking that training data is within the time window centred on test date
    -----------------------------------------------------------------------------
    
    #### Test date ==> 2017-01-13 21:00:00, window-length ==> 7 days #######
    count                     237
    unique                    237
    top       2017-01-07 20:00:00
    freq                        1
    first     2017-01-06 21:00:00
    last      2017-01-20 21:00:00
    Name: datetime_sent_hr, dtype: object


7.2 * Generate Event Freqs*
--------------------------------
The prediction is based on the most frequent event at the hour of interest over the moving window.


```python
# Power_state frequencies by the hour
hr_cnts = training_data.groupby(['hour_sent', 'power_state'])['power_state'].agg(['count'])
event_freqs = hr_cnts.reset_index()
```

7.3 * Return the event with most counts*
-------------------------------------


```python
pred_hr = prediction_date.hour
events_hr = event_freqs[event_freqs.hour_sent == pred_hr]
predicted_event = events_hr.max(axis=0)[clf.target_var]

print('Predicted [{}] vs. actual [{}] power-state'.format(predicted_event, actual))
```

    Predicted [1] vs. actual [1] power-state


8. * Computing evaluation metrics*
---------------------------------
*Demonstrate with one box*

- Precision
- Recall
- F1-score
- Accuracy


```python
predicted = []
actual = []
for idx, row in test_df.iterrows():
    test_date = row['datetime_sent_hr']
    predicted.append(clf.predict(prediction_date=test_date, box_id=test_box_id, target_loc=target_loc))
    actual.append(row['power_state'])

report = classification_report(y_pred=predicted, y_true=actual, digits=2)
print(report)
```

                 precision    recall  f1-score   support
    
              0       0.00      0.00      0.00        14
              1       0.94      1.00      0.97       236
    
    avg / total       0.89      0.94      0.92       250
    


    /usr/local/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)


8.1  * More detailed metrics available*
---------------------------------------


```python
results = cust_mod.nearest_neigbor_model_metrics_power_state(model_object=clf, test_data=test_df,
                                            box_id=test_box_id, xy=target_loc, target='power_state')
```


```python
print(results)
```

    Metric(support=250, actual_1=236, actual_0=14, accuracy=94.39999999999999, precision_1=94.39999999999999, recall_1=100.0, precision_0=nan, recall_0=0.0, tot_pred_0=0, tot_pred_1=250, correct_1=236, correct_0=0)


**EVALUATING OUT OF THE BOX MODEL**
=======================
==================================================================

1. 0 **Set up**
-----------------------------------
Lets consider the same test box:
We can select training data based on spatio-temporal window:
- No window: all of the data for all the time 
- Space-time window as in nearest neighbor: selected boxes (e.g., same box only)

When testing, we can compute metrics in 2 ways:
- Box based (other geographic region, box makes more sense)
- All data pooled togather

For this demonstration, we set up the test so that the situation is
the same as in previous model:
- take test data from a single box
- train model on selected boxes (based on neighbors)
- train model on all data

2.0 **Choose and build model**
----------------------------
For now we demonstrate wiht one model-Decision tree based model-ETC)

- Choose predicion features

- Set up train and test data (we use same test and train data as before)


```python
clf_etc = etc(n_estimators=100)
```


```python
# Set up training data
prediction_features = ['box_id', 'psu', 'lon', 'lat', 'hour_sent', 'month_sent', 
                       'day_sent', 'wk_day_sent','wk_end']

train_X = train_df[prediction_features].values
train_y = train_df['power_state'].values

clf_etc.fit(X=train_X, y=train_y)
```




    ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
               max_depth=None, max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
               oob_score=False, random_state=None, verbose=0, warm_start=False)



2.0 **Compute elavaluation metrics for out of box model**
--------------------------------------------------------
For now we demonstrate with one model-Decision tree based model-ETC)

- Choose predicion features

- Set up train and test data (we use same test and train data as before)


```python
# Lets predict the first event in test set
test_X = test_df[prediction_features].iloc[0].values
actual_0 = test_df.iloc[0].power_state
```


```python
predicted_etc = clf_etc.predict(test_X.reshape((1,9)))
```


```python
print('Predicted [{}] vs. actual [{}] power-state'.format(predicted_etc, actual_0))
```

    Predicted [[1]] vs. actual [1] power-state


3.0. **Computing evaluation metrics**


```python
X_test_events = test_df[prediction_features].values
y_actual = test_df['power_state'].values

class_report = classification_report(y_pred=clf_etc(X_test_events), y_true=y_actual)
print(class_report)
```

                 precision    recall  f1-score   support
    
              0       0.83      0.71      0.77        14
              1       0.98      0.99      0.99       236
    
    avg / total       0.97      0.98      0.98       250
    


*What if we train with data from neighboring boxes only to make things equal?*
-----------------------------------------------------------------------------


```python
# Set up training data
clf_neighbors = etc(n_estimators=100)

prediction_features = ['box_id', 'psu', 'lon', 'lat', 'hour_sent', 'month_sent', 
                       'day_sent', 'wk_day_sent','wk_end']

train_X = training_data[prediction_features].values
train_y = training_data['power_state'].values


clf_neighbors.fit(X=train_X, y=train_y)
```




    ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
               max_depth=None, max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
               oob_score=False, random_state=None, verbose=0, warm_start=False)




```python
X_test_events = test_df[prediction_features].values
y_actual = test_df['power_state'].values

class_report = classification_report(y_pred=clf_neighbors.predict(X_test_events), y_true=y_actual)
print(class_report)
```

                 precision    recall  f1-score   support
    
              0       0.04      0.07      0.05        14
              1       0.94      0.89      0.91       236
    
    avg / total       0.89      0.84      0.86       250
    


**So out of box model in this case seems to be doing better because of more data**
===================================================================================
