import xml.etree.ElementTree as ET
import csv
import time
import os


def xml_to_csv(xml_file, csv_file):
    '''
    Converts xml_file to csv file
    :param xml_file:
    :param csv_file:
    :return: A csv file
    '''
    try:
        e = ET.parse(xml_file).getroot()
        lst = e.findall('sms')
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
        print(data)


if __name__ == "__main__":
    file_name = 'sms_' + time.strftime('%m-%d-%Y') + '.csv'
    dir = "C:/Users/wb344850/WBG/William Hutchins Seitz - 01.data/raw_sms/"
    out_csv = dir + file_name
    xml_file = "C:/Users/wb344850/Google Drive/SMSBuckupRestore/sms.xml"