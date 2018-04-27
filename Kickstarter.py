# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:17:06 2018

@author: Patrick
"""

import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ks-projects-201801.csv', encoding = "UTF-8",)

df['campaign_len'] = pd.to_datetime(df['launched']) - pd.to_datetime(df['deadline'])
df['campaign_len'] = df['campaign_len'].apply(lambda x : str(x))
df['campaign_len'] = df['campaign_len'].apply(lambda x: x[:-15])
df['campaign_len'] = df['campaign_len'].apply(lambda x: abs(int(x)))

df[['pledged', 'state']].groupby(['state'], as_index = True).mean().sort_values(by = 'pledged', ascending = False)

df['goalqcut'] = pd.qcut(df['goal'], 6)
len(df[df['average_pledge'] == 0])/ len(df['average_pledge'])
df['average_pledge'] = df['pledged']/ df['backers']
df['average_pledge'].sort_values(ascending = True).head()

df['average_pledge'] = df['average_pledge'].fillna(0)
df['pledge_per_backer'] = df['average_pledge'].replace(np.inf, 0)
pldg_pr_bckr = df[['pledge_per_backer', 'main_category']].groupby(['main_category'], as_index = True).mean().sort_values(by = 'pledge_per_backer', ascending = False)
bckrs = df[['backers', 'main_category']].groupby(['main_category'], as_index = True).mean().sort_values(by = 'backers', ascending = False)
avg_pldg = df[['pledged', 'main_category']].groupby(['main_category'], as_index = True).mean().sort_values(by = 'pledged', ascending = False )

bckrs - pldg_pr_bckr

inter = [pldg_pr_bckr, bckrs, avg_pldg]
aurevoir = pd.concat(inter, axis = 1)
df['main_category'].value_counts()

backer_frame = pd.DataFrame(data = aurevoir, columns = ['pledge_per_backer', 'backers'])
backer_frame['pledge_backer_disparity'] = backer_frame['pledge_per_backer'] - backer_frame['backers']
backer_frame['pledge_backer_ratio'] = backer_frame['pledge_per_backer']/ backer_frame['backers']

backer_frame.sort_values(by = 'success_rate', ascending = False)


for target in targets:
    sns.distplot(target['pledged'])

sns.plt.show()

df['pledgeqcut'].value_counts()
    
low_pledge = df[df['pledged'] < 200]['main_category']
(low_pledge.value_counts() / df['main_category'].value_counts()).sort_values(ascending = False)
df['main_category'].value_counts()

low_pledge = df[df['goal'] < 200]['main_category']
(low_pledge.value_counts() / df['main_category'].value_counts()).sort_values(ascending = False)

df[['goal', 'main_category']].groupby(['main_category']).mean().sort_values(by = 'goal', ascending = False)
df['pledge_goal_ratio'] = df['pledged']/ df['goal']
df['pledged'] = df['pledged'].fillna(0)

low_pledge_success = df[df['pledged'] < 200 and df['state'] == 'successful']
low_pledge = df[df['pledged'] < 200]
low_pledge_success = low_pledge[low_pledge['state'] == 'successful']['main_category']
(low_pledge_success.value_counts() / df['main_category'].value_counts()).sort_values(ascending = False)

df['goal_disparity'] = df['pledged'] - df['goal']

df['goalcut'] = pd.cut(df['goal'], 6)
new_pledge = df[df['pledged'] < 10000]
new_pledge = new_pledge[new_pledge['goal'] < 10000]
new_pledge = new_pledge[new_pledge['pledged'] > 0]
sns.distplot(new_pledge['goal_disparity'])



new_pledge['goal_disparity'].value_counts()
pillar = []
cat = df[df['state'] == 'canceled']
for i in df['main_category'].unique():
    length = len(cat[cat['main_category'] == i])
    length2 = len(df[df['main_category'] == i])
    pillar.append(length/length2)

matic = []
dog = df[df['state'] == 'failed']
for i in df['main_category'].unique():
    length = len(dog[dog['main_category'] == i])
    length2 = len(df[df['main_category'] == i])
    matic.append(length/ length2)
big = []
for i in range(len(matic)):
    fish = pillar[i] / matic[i]
    big.append(fish)
    
boro = []
bird = df[df['state'] == 'successful']
for i in df['main_category'].unique():
    length = len(bird[bird['main_category'] == i])
    length2 = len(df[df['main_category'] == i])
    boro.append(length/ length2)

np.corrcoef(pillar, matic)[0][1]
np.corrcoef(pillar, boro)[0][1]
np.corrcoef(matic, boro)[0][1]

df['state'] = df['state'].map({'canceled': 0, 'failed': 0, 'successful': 1})
df = df.dropna(axis = 0, subset = ['state'])
df = df.dropna(axis = 0, subset = ['pledge_goal_ratio'])
binary_df['state'].mean()
success = binary_df[['main_category', 'state']].groupby(['main_category']).mean().sort_values(by = 'state', ascending = False)

backer_frame['success'] = success

np.corrcoef(backer_frame['success'], backer_frame['pledge_backer_ratio'])
df.head()
df2  = df[df['pledge_goal_ratio'] > 0]
corrmat = df.corr()['state']
success_corr = []
for i in df['main_category'].unique():
    corr = np.corrcoef(df2[df2['main_category'] == i]['state'], df2[df2['main_category'] == i]['pledge_goal_ratio'])[0][1]
    success_corr.append(corr)
    
success_corr = []
for i in df['main_category'].unique():
    corr = np.corrcoef(df2[df2['main_category'] == i]['state'], df2[df2['main_category'] == i]['pledge_backer_ratio'])[0][1]
    success_corr.append(corr)  
    print(success_corr)

succ_corr = pd.concat([df['main_category'].unique(), success_corr], axis = 1)
success_frame = pd.DataFrame([df['main_category'].unique(), success_corr], columns = ['categories', 'correlation'])

suc = pd.DataFrame({'category': df['main_category'].unique(), 'correlation': success_corr})
suc.sort_values(by = 'correlation', ascending = False)

df2['pledge_backer_ratio'] = df2['pledge_per_backer']/ df2['backers']
df2 = df2.dropna(axis = 0, subset = ['pledge_per_backer'])
len(df2)
df2.head()
df.head()
len(df2)
len(df)

df = df2
df.to_csv("Kickstart.csv", index = False)

len_count = []
for i in df['campaign_len'].unique():
    count = 0
    for j in range(len(df)):
        if df.iloc[j, 13] == i:
            count = count + 1
    len_count.append([i, count])
    
for i in df['campaign_len'].unique():  
    print(i)
cat = []
for i in range(len(df)):
    for j in range(len(len_count)):
       if df.iloc[i, 13] == len_count[j][0]:
           cat.append(len_count[j][1])

df.to_csv("Kickstart.csv", index = False)
df['count_campaign_len'] = cat
count_len_above_5000 = df[df['count_campaign_len'] > 5000]['state']

count_len_below_5000 = df[df['count_campaign_len'] < 5000]['state']

from scipy import stats

np.std(count_len_above_5000)
np.std(count_len_below_5000)

len(count_len_above_5000)
len(count_len_below_5000)

count_len_above_5000.mean()
count_len_below_5000.mean()

stats.ttest_ind(count_len_above_5000, count_len_below_5000, equal_var = False)

np.corrcoef(df['campaign_len'], df['goal'])

plt.plot(df[df['goal'] > 10000]['campaign_len'], df[df['goal'] > 10000]['goal'], 'ro')
plt.axis([0, 90, 10000, 1000000])
plt.show()

count_len_above_5000 = df[df['count_campaign_len'] > 5000]['goal']

count_len_below_5000 = df[df['count_campaign_len'] < 5000]['goal']

fuse = df['campaign_len'].isin([30, 45, 60])
fuse = fuse*1
df['304560'] = fuse

most_success = df[df['main_category'].isin(['Dance', 'Theater', 'Comics', 'Music', 'Art', 'Film & Video', 'Games'])]['304560']

least_success = df[df['main_category'] not in ['Dance', 'Theater', 'Comics', 'Music', 'Art', 'Film & Video', 'Games']]['304560']

hope = df['main_category'].isin(['Dance', 'Theater', 'Comics', 'Music', 'Art', 'Film & Video', 'Games'])
here = hope == False
least_success = df[here]['304560']

np.std(most_success)
np.std(least_success)

len(most_success)
len(least_success)

stats.ttest_ind(most_success, least_success, equal_var = False)

df.to_csv("Kickstart.csv", index = False)




