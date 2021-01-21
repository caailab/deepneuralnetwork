# coding: utf8
_author_ = 'Licheng QU'

import os
import numpy as np
import pandas as pd


def df_stamp(df):
    str = "{:0.0f}/{:0.0f}/{:0.0f} {:0.0f}:{:0.0f}".format(df['year'], df['month'], df['day'], df['hour'], df['minute'])
    s = pd.to_datetime(str, format="%Y/%m/%d %H:%M")
    return s


def df_date(stamp):
    # d = stamp[:10]
    d = pd.to_datetime(stamp).strftime("%Y-%m-%d")
    return d


def df_year(stamp):
    # y = stamp[:4]
    y = pd.to_datetime(stamp).year
    return int(y)


def df_month(stamp):
    # m = stamp[5:7]
    m = pd.to_datetime(stamp).month
    return int(m)


def df_day(stamp):
    # d = stamp[8:10]
    d = pd.to_datetime(stamp).day
    return int(d)


def df_hour(stamp):
    # h = stamp[11:13]
    h = pd.to_datetime(stamp).hour
    return (int(h))


def df_minute(stamp):
    #m = stamp[14:16]
    m = pd.to_datetime(stamp).minute
    return (int(m))


def df_weekday(stamp):
    w = pd.to_datetime(stamp).dayofweek     # Monday is 0 and Sunday is 6
    return (int(w)+1)                       # Monday is 1 and Sunday is 7


def df_holiday(stamp):
    dt = pd.to_datetime(stamp)

    m = dt.month
    d = dt.day
    w = dt.dayofweek    # The day of the week with Monday=0, Sunday=6
    w += 1              # Monday is 1 and Sunday is 7

    MONDAY = 1
    THURSDAY = 4
    SUNDAY = 7

    SPECIAL_DAY = 0.0
    IMPORTANT_DAY = 0.0
    NATIONAL_DAY = 0.5
    holiday = 0.0

    # New Year's Day
    if ( m == 1 and d == 1):
        holiday = NATIONAL_DAY
    # Martin Luther King, Jr. Day
    elif (m == 1 and w == MONDAY and ((d - w) // 7) == 2):
        holiday = NATIONAL_DAY
    # Valentine's Day
    elif ( m == 2 and d == 14):
        holiday = IMPORTANT_DAY
    # President's Day
    elif (m == 2 and w == MONDAY and ((d - w) // 7) == 2):
        holiday = NATIONAL_DAY
    # Sant Patrick's Day
    elif (m == 3 and d == 17):
        holiday = SPECIAL_DAY
    # April Fool's Day
    elif (m == 4 and d == 1):
        holiday = SPECIAL_DAY
    # Mother's Day
    elif (m == 5 and w == SUNDAY and ((d - w) // 7) == 1):
        holiday = IMPORTANT_DAY
    # Memorial Day
    elif (m == 5 and w == MONDAY and (31-d) < 7 ):
        holiday = NATIONAL_DAY
    # Father's Day
    elif (m == 6 and w == SUNDAY and ((d - w) // 7) == 2):
        holiday = IMPORTANT_DAY
    # Independence Day
    elif ( m == 7 and d == 4):
        holiday = NATIONAL_DAY
    # Labor Day
    elif (m == 9 and w == MONDAY and ((d - w) // 7) == 0):
        holiday = NATIONAL_DAY
    # Columbus Day
    elif (m == 10 and w == MONDAY and ((d - w) // 7) == 1):
        holiday = NATIONAL_DAY
    # Halloween
    if (m == 10 and d == 31):
        holiday = SPECIAL_DAY
    # All Saints' Day
    if (m == 11 and d == 1):
        holiday = IMPORTANT_DAY
    # Veterans Day
    if ( m == 11 and d == 11):
        holiday = NATIONAL_DAY
    # Thanksgiving Day
    elif (m == 11 and w == THURSDAY and ((d - w) // 7) == 3):
        holiday = NATIONAL_DAY
    # Christmas Eve
    elif (m == 12 and d == 24):
        holiday = IMPORTANT_DAY
    # Christmas
    elif (m == 12 and d == 25):
        holiday = NATIONAL_DAY

    # print("{}/{} week {}  holiday {} == {}".format(m, d, w, holiday, (d - 7) // 7))

    return holiday


# daylight saving time
def df_daylight(stamp):
    dt = stamp#[:10]
    dt = pd.to_datetime(stamp)

    m = dt.month
    d = dt.day
    w = dt.dayofweek    #The day of the week with Monday=, Sunday=6
    w += 1

    SUNDAY = 7

    # First Sunday of November
    DAYLIGHT_SAVING_TIME_WINTER = 0.3
    # Second Sunday of March
    DAYLIGHT_SAVING_TIME_SUMMER = 0.5

    daylight = 0

    if ( m < 3 or m > 11):
        daylight = DAYLIGHT_SAVING_TIME_WINTER
    elif (m == 3):
        if (((d - w) // 7) >= 1 or (((d - w) // 7) == 0 and w ==SUNDAY)):
            daylight = DAYLIGHT_SAVING_TIME_SUMMER
        else:
            daylight = DAYLIGHT_SAVING_TIME_WINTER
    elif (m == 11):
        if (((d - w) // 7) >= 0 or (((d - w) // 7) == -1 and w ==SUNDAY)):
            daylight = DAYLIGHT_SAVING_TIME_WINTER
        else:
            daylight = DAYLIGHT_SAVING_TIME_SUMMER
    else:
        daylight = DAYLIGHT_SAVING_TIME_SUMMER

    # print("{}/{} week {}  daylight {} == {}".format(m, d, w, daylight, (d - w) // 7))

    return daylight


def load_traffic_data_set_stamp(csvfile):
    '''
    Load the 5-minute traffic data,
    then splite the stamp field into year, month, day, weekday, hour and minute.

    :param csvfile: 5-minute traffic data
    :return: features and labels
    '''

    df = pd.read_csv(csvfile, header=None)
    df.columns = ['name', 'stamp', 'speed', 'volume']
    df['year'] = df.stamp.apply(df_year)
    df['month'] = df.stamp.apply(df_month)
    df['day'] = df.stamp.apply(df_day)
    df['weekday'] = df.stamp.apply(df_weekday)
    df['hour'] = df.stamp.apply(df_hour)
    df['minute'] = df.stamp.apply(df_minute)
    # print('Traffic Data Set', df.shape)
    # print(df)

    features = np.array(df.loc[:, ['year', 'month', 'day', 'weekday', 'hour', 'minute']], np.float32)

    labels = np.array(df.loc[:, ['speed', 'volume']], np.float32)

    print('Traffic Data Set :', features.shape, labels.shape)
    # print('Traffic set', features, labels)

    return features, labels


def load_traffic_data(csvfile):
    '''
    Load traffic data with full field (don't need to convert the stamp field).
    Then add a holiday field.

    :param csvfile: x-minute traffic data
    :return: features and labels
    '''

    df = pd.read_csv(csvfile, header=None)
    df.columns = ['name', 'stamp', 'speed', 'volume', 'year', 'month', 'day', 'weekday', 'hour', 'minute', 'rain']
    df['holiday'] = df.stamp.apply(df_holiday)
    df['daylight'] = 0.0 #df.stamp.apply(df_daylight)
    # print('Traffic Data Set :', df.shape)
    # print(df)

    features = np.array(df.loc[:, ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'holiday', 'daylight', 'rain']], np.float32)

    labels = np.array(df.loc[:, ['volume']], np.float32)

    print('Traffic Data Set :', features.shape, labels.shape)
    # print('Traffic set', features, labels)

    return features, labels


def load_traffic_data_cache(csvfile):
    '''
    Load traffic data with full field (stamp field has been converted before).
    Then add a holiday field.

    :param csvfile: x-minute traffic data
    :return: features, labels and stamp
    '''

    df = pd.read_csv(csvfile, header=1, parse_dates=[0])
    df.columns = ['stamp', 'year', 'month', 'day', 'hour', 'minute', 'weekday', 'holiday', 'daylight', 'rain', 'volume']
    # print('Traffic Data Set :', df.shape)
    # print(df)

    features = np.array(df.loc[:, ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'holiday', 'daylight', 'rain']], np.float32)
    labels = np.array(df.loc[:, ['volume']], np.float32)
    stamp = np.array(df.loc[:, ['stamp']])

    print('Traffic Data Set :', features.shape, labels.shape, stamp.shape)
    # print('Traffic set', features, labels)

    return features, labels, stamp


def load_traffic_data_resample(csvfile, interval=5, minvolume=2):
    '''
    Load the 5-minute traffic data and resample it according to the interval.
    Splite the stamp field into year, month, day, weekday, hour and minute.
    And save the resample result into csvfile-{interval}minute.csv.
    If there is a cached data file generated before, the function will return it immediately.

    @param csvfile: 5-minute traffic data
    @param interval: resample interval
    @param  minvolume: minimum volume per minute
    @return: features, labels and stamp
    '''

    cache_csv = csvfile[:-4] + '-{:02}min.csv'.format(interval)
    if os.access(cache_csv, os.R_OK):
        return load_traffic_data_cache(cache_csv)

    period = str(interval) + 'T'
    df = pd.read_csv(csvfile, parse_dates=[0])   #, header=None)
    df.columns = ['stamp', 'volume']

    # df['stamp'] = df['stamp'].astype('datetime64[ns]')
    # df.set_index('stamp', drop=False, inplace=True)
    # df = df.resample('10T', how='sum')
    time_list = pd.to_datetime(df['stamp'])    # transform column into datetime type
    time_series = df.set_index(time_list)      # set time serials as row index
    if pd.__version__ >= "0.18.0":
        df = time_series.resample(period).sum()
    else:
        df = time_series.resample(period, how='sum')

    # df = df.loc[df['hour'] >= 6]
    # df = df.loc[df['hour'] < 22]
    df = df.loc[df['volume'] >= minvolume * interval]

    df['stamp'] = df.index
    df['year'] = df.stamp.apply(df_year)
    df['month'] = df.stamp.apply(df_month)
    df['day'] = df.stamp.apply(df_day)
    df['weekday'] = df.stamp.apply(df_weekday)
    df['hour'] = df.stamp.apply(df_hour)
    df['minute'] = df.stamp.apply(df_minute)
    df['holiday'] = df.stamp.apply(df_holiday)
    df['daylight'] = 0.0 #df.stamp.apply(df_daylight)
    df['rain'] = 0.0
    del df['stamp']

    print('Traffic Data Set :', df.shape)
    print(df)
    df.to_csv(cache_csv, columns=['year', 'month', 'day', 'hour', 'minute', 'weekday', 'holiday', 'daylight', 'rain', 'volume'], index=True)

    features = np.array(df.loc[:, ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'holiday', 'daylight', 'rain']], np.float32)
    labels = np.array(df.loc[:, ['volume']], np.float32)
    stamp = np.array(df.index, np.datetime64)

    print('Traffic Data Set :', features.shape, labels.shape, stamp.shape)
    # print('Traffic set', features, labels)

    return features, labels, stamp


def load_and_clean_traffic_data(csvfile):
    '''
    Load and clean the 5-minute traffic data
    :param csvfile: 5-minute traffic data
    :return: features and labels
    '''

    df = pd.read_csv(csvfile)   #, header=None)
    df.columns = ['stamp', 'volume']
    df['year'] = df.stamp.apply(df_year)
    df['month'] = df.stamp.apply(df_month)
    df['day'] = df.stamp.apply(df_day)
    df['weekday'] = df.stamp.apply(df_weekday)
    df['hour'] = df.stamp.apply(df_hour)
    df['minute'] = df.stamp.apply(df_minute)
    df['holiday'] = df.stamp.apply(df_holiday)
    df['daylight'] = 0.0 #df.stamp.apply(df_daylight)
    df['rain'] = 0.0
    df['date'] = df.stamp.apply(df_date)
    df['timepoint'] = df['hour'] * 12 + (df['minute'] // 5) + 1
    # print('Traffic Data Set :', df.shape)
    # print(df)

    # df_g = df.groupby(['year', 'month', 'day', 'weekday', 'hour', 'minute'], sort=False).sum()
    # df_g.to_csv('123_hour.csv', index=False)
    groupby = df.groupby(['date'], as_index=False)['volume'].size() #count()
    # print(groupby)

    dfg = groupby.reset_index()#dfc.to_frame #.agg([('M', 'mean')])
    dfg.columns = ['date', 'size']
    dfg = dfg.loc[dfg['size'] == 288]
    # print("dfg", dfg.shape)
    # print(dfg)

    df = df.merge(dfg, on='date')
    # print("df merge, shape=", df.shape)
    # print(df)

    return df


def load_traffic_data_clean(csvfile, interval):
    '''
    Clean and save the 5-minute traffic data.
    if the cache data exists then read the cached csv file directly;
    if the cache data does not exist then
    :param csvfile: 5-minute traffic data
    :return: features and labels
    '''

    cache_csv = csvfile[:-4] + '-{}minute-clean.csv'.format(interval)
    if os.access(cache_csv, os.R_OK):
        return load_traffic_data_cache(cache_csv)

    df = load_and_clean_traffic_data(csvfile)
    df['minute'] = (df['minute'] // interval) * interval
    # df = df.loc[df['hour'] >= 6]
    # df = df.loc[df['hour'] < 22]
    df = df.loc[df['volume'] >= 30]
    # print('Traffic Data Set :', df.shape)
    # print(df)

    # df_g = df.groupby(['year', 'month', 'day', 'weekday', 'hour', 'minute'], sort=False).sum()
    # df_g.to_csv('123_hour.csv', index=False)
    groupby = df.groupby(['year', 'month', 'day', 'hour', 'minute', 'weekday', 'holiday', 'daylight', 'rain'], as_index=False)['volume'].sum()
    # print(groupby)

    df = groupby.reset_index()
    # regenerate the stamp column
    df['stamp'] = df.apply(df_stamp, axis=1)

    print(interval, " minute data", df.shape)
    print(df)

    df.to_csv(cache_csv, columns=['stamp', 'year', 'month', 'day', 'hour', 'minute', 'weekday', 'holiday', 'daylight', 'rain', 'volume'], index=False)

    features = np.array(df.loc[:, ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'holiday', 'daylight', 'rain']], np.float32)
    labels = np.array(df.loc[:, ['volume']], np.float32)
    stamp = np.array(df.loc[:, ['stamp']], np.datetime64)

    print('Traffic Data Set :', features.shape, labels.shape)
    print('Volume MAX : {}, MIN : {}'.format(labels.max(), labels.min()))
    # print('Traffic set', features, labels)

    return features, labels, stamp


def normalize_data(features, labels, factor):
    features[:, 0] -= 2000.0
    features[:, 0] /= 1000.0    # year
    features[:, 1] /= 100.0     # month
    features[:, 2] /= 100.0     # day
    features[:, 3] -= 12
    features[:, 3] /= 15.0      # hour
    features[:, 4] -= 30.0
    features[:, 4] /= 40.0      # minute
    features[:, 5] -= 3.5
    features[:, 5] /= 10.0      # weekday
    # features[:, 6] -= 0.0
    # features[:, 6] /= 1        # holiday
    # features[:, 7] -= 0.0
    # features[:, 7] /= 1        # daylight
    features[:, 8] /= 100.0     # rain

    # factor /= 2
    # labels -= factor
    labels /= factor            # volume

    for i in range(1, len(labels)-2):
        labels[i] = labels[i-1] * 0.3 + labels[i] * 0.4 + labels[i+1] * 0.3
    
    return features, labels


def unnormalize_data(labels, factor):
    # factor /= 2
    labels *= factor
    # labels += factor  # volume

    return labels


def smooth_data(labels):
    for i in range(1, len(labels)-2):
        labels[i] = labels[i-1] * 0.2 + labels[i] * 0.6 + labels[i+1] * 0.2

    return labels


def clean_traffic_data(csvfile):
    '''
    delete the incomplete data
    '''
    df = pd.read_csv(csvfile, header=None)
    df.columns = ['name', 'stamp', 'speed', 'volume', 'year', 'month', 'day', 'weekday', 'hour', 'minute', 'rain']
    df['holiday'] = df.stamp.apply(df_holiday)
    df['daylight'] = 0.0 #df.stamp.apply(df_daylight)
    df['date'] = df.stamp.apply(df_date)
    df['timepoint'] = df['hour'] * 12 + (df['minute'] // 5) + 1
    # print('Traffic Data Set :', df.shape)
    # print(df)

    groupby = df.groupby(['date'], as_index=False)['volume'].size() #count()
    # print(groupby)

    dfg = groupby.reset_index()#dfc.to_frame #.agg([('M', 'mean')])
    dfg.columns = ['date', 'size']
    dfg = dfg.loc[dfg['size'] == 288]
    # print("dfg", dfg.shape)
    # print(dfg)

    df = df.merge(dfg, on='date')
    # print("df merge, shape=", df.shape)
    # print(df)

    features = np.array(df.loc[:, ['year', 'month', 'day', 'hour', 'minute', 'weekday', 'holiday', 'daylight', 'rain']], np.float32)

    labels = np.array(df.loc[:, ['speed', 'volume']], np.float32)

    print('Traffic Data Set :', features.shape, labels.shape)
    # print('Traffic set', features, labels)

    return features, labels

def traffic_data_to_daily(csvfile):
    '''
    pivot the traffic data to one day per line
    and save the new data to ~.pivot.csv
    '''
    df = pd.read_csv(csvfile, header=None)
    df.columns = ['name', 'stamp', 'speed', 'volume', 'year', 'month', 'day', 'weekday', 'hour', 'minute', 'rain']

    df['date'] = df.stamp.apply(df_date)
    df['timepoint'] = df['hour'] * 12 + (df['minute'] // 5) + 1
    # print('Traffic Data Set :', df.shape)
    # print(df)

    dfp = df.pivot('date', 'timepoint', 'volume')
    # dfp = dfp.fillna(0.0)
    dfp = dfp.dropna()
    print(dfp)
    dfp.to_csv(csvfile[:-4] + '.pivot.csv')

    labels = np.array(dfp.loc[:, :], np.float32)

    print('Daily Traffic Data Set Shape:', labels.shape)
    # print('Daily Traffic set', features, labels)

    return labels


if __name__ == '__main__':
    """ Generate resampled csv files
    """

    # csvfile = '../../dataset/005es18066/18066-I-201603.csv'
    # load_traffic_data_resample(csvfile, 5)

    #              ../dataset/volume-005es18017-I-2015.csv
    csvfilename = '../dataset/volume-005es{}-I-{}.csv'
    yearmonth = ('2015', '201601', '201602', '201603')
    intervals = (5, 10, 15, 20, 30, 60)
    mileposts = ('18017', '18066', '18115', '18204', '18264', '18322', '18449', '18507', '18548', '18635', '18707', '18739', '18797', '18846', '18900', '18998')

    for m in mileposts:
        for y in yearmonth:
            csvfile = csvfilename.format(m, y)
            print(csvfile)
            for i in intervals:
                # resample csv file
                load_traffic_data_resample(csvfile, i)

