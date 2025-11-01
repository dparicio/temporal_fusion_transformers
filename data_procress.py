import numpy as np
import pandas as pd

df = pd.read_csv("LD2011_2014.txt", index_col=0, sep=';', decimal=',')
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)

# Used to determine the start and end dates of a series
output = df.resample('1h').mean().replace(0., np.nan)

earliest_time = output.index.min()

df_list = []
for label in output:
    print('Processing {}'.format(label))
    srs = output[label]

    start_date = min(srs.fillna(method='ffill').dropna().index)
    end_date = max(srs.fillna(method='bfill').dropna().index)

    active_range = (srs.index >= start_date) & (srs.index <= end_date)
    srs = srs[active_range].fillna(0.)

    tmp = pd.DataFrame({'power_usage': srs})
    date = tmp.index
    tmp['t'] = (date - earliest_time).seconds / 60 / 60 + (
        date - earliest_time).days * 24
    tmp['days_from_start'] = (date - earliest_time).days
    tmp['categorical_id'] = label
    tmp['date'] = date
    tmp['id'] = label
    tmp['hour'] = date.hour
    tmp['day'] = date.day
    tmp['day_of_week'] = date.dayofweek
    tmp['month'] = date.month

    df_list.append(tmp)

output = pd.concat(df_list, axis=0, join='outer').reset_index(drop=True)

output['categorical_id'] = output['id'].copy()
output['hours_from_start'] = output['t']
output['categorical_day_of_week'] = output['day_of_week'].copy()
output['categorical_hour'] = output['hour'].copy()

# Filter to match range used by other academic papers
output = output[(output['days_from_start'] >= 1096)
                & (output['days_from_start'] < 1346)].copy()


output.to_csv("processed_power_usage.csv", index=False)

print("Debug")
