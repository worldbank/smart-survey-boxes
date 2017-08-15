"""
Automated tests to make debugging easy
"""

from pypower import data_utils as ut
from datetime import datetime
import unittest

class TestDataUtils(unittest.TestCase):
    # ------------SET UP WORKING DIRECTORY-----------------
    platform = 'mac'
    data_dir_mac = "/Users/dmatekenya/Google Drive/World-Bank/electricity_monitoring/01.data/"
    outputs_dir_mac = "/Users/dmatekenya/Google Drive/World-Bank/electricity_monitoring/05.outputs/"

    if platform == 'mac':
        data_dir = data_dir_mac

    box_file = data_dir + 'Boxes.csv'
    sms_xml = data_dir + 'sms.xml'

    # ------------ GET BOX METADATA --------------------
    box_metadata = ut.box_loc_metadata_as_dict(box_file)

    def test_number_of_boxes_generated(self):
        created_objects = ut.create_box_obj_from_events(self.sms_xml, self.box_metadata, after_event_threshold=12,
                                                        before_event_threshold=12, how_to_insert='prev')
        correct_number = 296

        self.assertGreaterEqual(len(created_objects), correct_number, 'Not all box data generated')

    #test if datetime_sent_hr isnt null
    def test_box_objects_events_contain_date_sent(self):
        # capture the results of the function
        created_objects = ut.create_box_obj_from_events(self.sms_xml, self.box_metadata, after_event_threshold=12,
                                                        before_event_threshold=12, how_to_insert='prev')
        for box in list(created_objects.values()):
            events = box.hourly_events
            # check event properties
            for ev in list(events.values()):
                ev_data = ev.get_selected_event_metadata()

                #check if datetime_sent_hr is not None
                self.assertIn('datetime_sent_hr', ev_data)

    def test_box_objects_events_contain_datesent_not_null(self):
        # capture the results of the function
        created_objects = ut.create_box_obj_from_events(self.sms_xml, self.box_metadata, after_event_threshold=12,
                                                        before_event_threshold=12, how_to_insert='prev')
        for box in list(created_objects.values()):
            events = box.hourly_events
            # check event properties
            for ev in list(events.values()):
                ev_data = ev.get_selected_event_metadata()

                # get datesent
                date_sent = ev_data.get('datetime_sent_hr')

                #check if datetime_sent_hr is not None
                self.assertIsNotNone(date_sent, 'Date-sent is null')

if __name__ == "__main__":
    unittest.main()

