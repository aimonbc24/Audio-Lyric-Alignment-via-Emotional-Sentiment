import DALI as dali_code
import os

absolute_path = os.path.dirname(os.path.abspath(__file__))
dali_data_path = absolute_path + '/data/'

dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=[])
dali_info = dali_code.get_info(dali_data_path + 'info/DALI_DATA_INFO.gz')
# print(dali_info])
# print(dali_info[0])

count = 0
for entry in dali_data:
    count += 1
    print(entry)
    if count > 10:
        break