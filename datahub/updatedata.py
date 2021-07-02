import os
from mtools import list_mask

def get_months_dates(data_path):
    months = os.listdir(data_path)
    months = list_mask(months, [month.startswith('2') for month in months])
    months.sort()
    filenames = [os.listdir(os.path.join(data_path, month)) for month in months]
    data_names = [[filename[:6] for filename in filenames_] for filenames_ in filenames]
    dates = [list(set(data_names_)) for data_names_ in data_names]
    dates = [list_mask(dates_, [date.startswith('2') for date in dates_]) for dates_ in dates]
    for dates_ in dates:
        dates_.sort()
    return months, dates