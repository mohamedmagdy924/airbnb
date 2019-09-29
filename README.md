

```python
import warnings
warnings.filterwarnings('ignore')

```

# <center>Mohamed Abbas</center>
### <center>mohamed_magdy92@live.com</center>

## 1. Introduction

Founded in 2008, Airbnb’s mission is to create a world where people can belong through healthy travel that is local, authentic, diverse, inclusive and sustainable. Airbnb uniquely leverages technology to economically empower millions of people around the world to unlock and monetize their spaces, passions and talents and become hospitality entrepreneurs. Airbnb’s accommodation marketplace provides access to 7 million unique places to stay in more than 100,000 cities and 191 countries and regions. With Experiences, Airbnb offers unprecedented access to local communities and interests through 40,000 unique, handcrafted activities run by hosts across 1,000+ cities around the world. Airbnb’s people-to-people platform benefits all our stakeholders, including hosts, guests, employees and the communities in which we operate.

## 2. Dataset

### Description of the Dataset
In this challenge, I am given a list of users along with their demographics, web session records, and some summary statistics. You are asked to predict which country a new user's first booking destination will be. All the users in this dataset are from the USA.

There are 12 possible outcomes of the destination country: 'US', 'FR', 'CA', 'GB', 'ES', 'IT', 'PT', 'NL','DE', 'AU', 'NDF' (no destination found), and 'other'. Please note that 'NDF' is different from 'other' because 'other' means there was a booking, but is to a country not included in the list, while 'NDF' means there wasn't a booking.

The training and test sets are split by dates. In the test set, you will predict all the new users with first activities after 7/1/2014 (note: this is updated on 12/5/15 when the competition restarted). In the sessions dataset, the data only dates back to 1/1/2014, while the users dataset dates back to 2010. 

### User Dataset

1. __train_users.csv__ - The training set of users
2. __test_users.csv__ - the test set of users

   - id: user id
   - date_account_created: the date of account creation
   - timestamp_first_active: timestamp of the first activity, note that it can be earlier than date_account_created or date_first_booking because a user can search before signing up
   - date_first_booking: date of first booking
   - gender
   - age
   - signup_method
   - signup_flow: the page a user came to signup up from
   - language: international language preference
   - affiliate_channel: what kind of paid marketing
   - affiliate_provider: where the marketing is e.g. google, craigslist, other
   - first_affiliate_tracked: whats the first marketing the user interacted with before the signing up
   - signup_app
   - first_device_type
   - first_browser
   - country_destination: this is the target variable you are to predict

### Session Dataset
1. __sessions.csv__ - web sessions log for users
  - user_id: to be joined with the column 'id' in users table
  - action
  - action_type
  - action_detail
  - device_type
  - secs_elapsed

### Countries Dataset
1. __countries.csv__ - Summary statistics of destination countries in this dataset and their locations
2. __age_gender_bkts.csv__ - Summary statistics of users' age group, gender, country of destination


# importing needed pacakges


```python
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import sys 
import os 
import folium
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from IPython.display import HTML, display
from sklearn.decomposition import PCA

```

## reading csv files

### removing "NDF" destination 


```python
## loading training csv file for exploration and preprocessing 
train_df=pd.read_csv("airbnb/train_users_2.csv")  ## Training dataframe
train_df=train_df.loc[train_df.country_destination!="NDF"]
```

###  displaying data


```python
display(train_df.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date_account_created</th>
      <th>timestamp_first_active</th>
      <th>date_first_booking</th>
      <th>gender</th>
      <th>age</th>
      <th>signup_method</th>
      <th>signup_flow</th>
      <th>language</th>
      <th>affiliate_channel</th>
      <th>affiliate_provider</th>
      <th>first_affiliate_tracked</th>
      <th>signup_app</th>
      <th>first_device_type</th>
      <th>first_browser</th>
      <th>country_destination</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>4ft3gnwmtx</td>
      <td>2010-09-28</td>
      <td>20090609231247</td>
      <td>2010-08-02</td>
      <td>FEMALE</td>
      <td>56.0</td>
      <td>basic</td>
      <td>3</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>untracked</td>
      <td>Web</td>
      <td>Windows Desktop</td>
      <td>IE</td>
      <td>US</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bjjt8pjhuk</td>
      <td>2011-12-05</td>
      <td>20091031060129</td>
      <td>2012-09-08</td>
      <td>FEMALE</td>
      <td>42.0</td>
      <td>facebook</td>
      <td>0</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>untracked</td>
      <td>Web</td>
      <td>Mac Desktop</td>
      <td>Firefox</td>
      <td>other</td>
    </tr>
    <tr>
      <th>4</th>
      <td>87mebub9p4</td>
      <td>2010-09-14</td>
      <td>20091208061105</td>
      <td>2010-02-18</td>
      <td>-unknown-</td>
      <td>41.0</td>
      <td>basic</td>
      <td>0</td>
      <td>en</td>
      <td>direct</td>
      <td>direct</td>
      <td>untracked</td>
      <td>Web</td>
      <td>Mac Desktop</td>
      <td>Chrome</td>
      <td>US</td>
    </tr>
    <tr>
      <th>5</th>
      <td>osr2jwljor</td>
      <td>2010-01-01</td>
      <td>20100101215619</td>
      <td>2010-01-02</td>
      <td>-unknown-</td>
      <td>NaN</td>
      <td>basic</td>
      <td>0</td>
      <td>en</td>
      <td>other</td>
      <td>other</td>
      <td>omg</td>
      <td>Web</td>
      <td>Mac Desktop</td>
      <td>Chrome</td>
      <td>US</td>
    </tr>
    <tr>
      <th>6</th>
      <td>lsw9q7uk0j</td>
      <td>2010-01-02</td>
      <td>20100102012558</td>
      <td>2010-01-05</td>
      <td>FEMALE</td>
      <td>46.0</td>
      <td>basic</td>
      <td>0</td>
      <td>en</td>
      <td>other</td>
      <td>craigslist</td>
      <td>untracked</td>
      <td>Web</td>
      <td>Mac Desktop</td>
      <td>Safari</td>
      <td>US</td>
    </tr>
  </tbody>
</table>
</div>


### Replacing -unknown- values with nan
after displaying data i found that there is -unknown- values in the categorical variables that need to be replaced 


```python
## replace unknown value with nan
train_df.replace('-unknown-', np.nan, inplace=True)
```

### Displaying datatypes 


```python
display(train_df.dtypes)
```


    id                          object
    date_account_created        object
    timestamp_first_active       int64
    date_first_booking          object
    gender                      object
    age                        float64
    signup_method               object
    signup_flow                  int64
    language                    object
    affiliate_channel           object
    affiliate_provider          object
    first_affiliate_tracked     object
    signup_app                  object
    first_device_type           object
    first_browser               object
    country_destination         object
    dtype: object


## Visualization  

### Age Visualization 


```python
plt.figure(figsize=(20,10))
sns.distplot(train_df.age.dropna())
plt.xlabel("Age",fontsize="x-large")
plt.title("Age Distrubution")
plt.xticks(fontsize="large")
plt.savefig("output.png")
plt.show()
```


![png](output_15_0.png)


### Age data
after visualizing the age data i found that the age needs more adjustment before completing the visualization , so i started displaying more statitcs for the age and i found that the age min is 2 and max is 2014 which can't be true . 


```python
display(train_df.age.describe())
```


    count    68532.000000
    mean        47.872629
    std        146.042716
    min          2.000000
    25%         28.000000
    50%         33.000000
    75%         42.000000
    max       2014.000000
    Name: age, dtype: float64


we found some data that could be enterted by wrong year of birth instead of age this can be handeled easily
subtrut dates between 1915 and 1997 from 2015 age limit (18-100)
below 18 or over 100 replaced with nan


```python
av=train_df.age.values
train_df['age'] = np.where(np.logical_and(av>1915, av<1997), 2015-av, av)
train_df['age'] = np.where(np.logical_or(av<18, av>100), np.nan, av)
```

distplot to get better look after age adjustment


```python
### some data visualiztion to get better look after age adjustment
plt.figure(figsize=(20,10))
sns.distplot(train_df.age.dropna())
plt.xlabel("Age",fontsize="x-large")
plt.title("Age Distrubution")
plt.xticks(fontsize="large")
plt.savefig("output1.png")
plt.show()
```


![png](output_21_0.png)


#### common age for travellers is between 20 and 45 

## Gender 

first we need to check the null values in gender and replace it with unknown


```python
### now let's look about the Gender 
print(train_df.gender.isnull().sum())
train_df.gender=train_df.gender.fillna("unknown")
```

    29018
    

Visualizing the data would give us better look to the ratio between genders 


```python
explode = (0.1, 0, 0,0)
plt.figure(figsize=(20,10))
plt.pie(x=train_df.gender.value_counts(),labels=("unknown","female","male","other"),autopct='%.2f',explode=explode,shadow=True)
plt.title("Sign_up_gender",fontsize="xx-large")
plt.savefig("Gender.png")
plt.show()

```


![png](output_27_0.png)



```python
plt.figure(figsize=(20,10))
sns.countplot(x="gender",data=train_df)
plt.xlabel("Gender",fontsize="x-large")
plt.ylabel("Counts",fontsize="x-large")
plt.title("Gender")
plt.xticks(fontsize="large")
plt.yticks(fontsize="large")
plt.savefig("output2.png")
plt.show()
```


![png](output_28_0.png)


## Country

by plotting the country_destination i found that the most visited country is US


```python
plt.figure(figsize=(20,10))
sns.countplot(x="country_destination",data=train_df)
plt.xlabel("Countries",fontsize="x-large")
plt.ylabel("Numbers of visitors",fontsize="x-large")
plt.title("Countries")
plt.xticks(fontsize="large")
plt.yticks(fontsize="large")
plt.savefig("output3.png")
```


![png](output_30_0.png)


## Checking country with Gender
now i will check the gender prefrence for country_destination 


```python
### check gender with countries 
plt.figure(figsize=(20,10))
sns.countplot(x="country_destination",data=train_df,hue="gender")
plt.xlabel("Countries",fontsize="x-large")
plt.ylabel("Gender",fontsize="x-large")
plt.title("Gender vs Counties")
plt.xticks(fontsize="large")
plt.yticks(fontsize="large")
plt.legend(fontsize="x-large")
plt.savefig("output4.png")

```


![png](output_32_0.png)


#### We can't have a specific prefered destination depending on the gender because as we see the ratios are so close

## Signup_app vs Country
Here we will try again to find relation between signup_app and the country_destination


```python
### check signup_app with countries 
plt.figure(figsize=(20,10))
sns.countplot(x="country_destination",data=train_df,hue="signup_app")
plt.xlabel("Countries",fontsize="x-large")
plt.ylabel("signup_app",fontsize="x-large")
plt.title("signup_app vs Counties")
plt.xticks(fontsize="large")
plt.yticks(fontsize="large")
plt.legend(fontsize="x-large")
plt.savefig("output6.png")

```


![png](output_35_0.png)


Again we can't find a relation between the Signup_app and the Country except that most of users prefer to signup throught web

### Age vs Signup_app vs Country



```python
### check age with countries with signup_app 
plt.figure(figsize=(30,20))
sns.catplot(x="country_destination",y="age",data=train_df,hue="signup_app",kind="strip",ci=None,col="signup_app",col_wrap=2,sharex=False,sharey=False)
plt.xlabel("Countries",fontsize="x-large")
plt.ylabel("Age",fontsize="x-large")
plt.title("Age vs Counties vs signup_app")
plt.xticks(fontsize="large")
plt.yticks(fontsize="large")
plt.legend(fontsize="x-large")
plt.savefig("output5.png")

```


    <Figure size 2160x1440 with 0 Axes>



![png](output_38_1.png)


#### here we can find that most of users who are over 70 years old and using browser as there signup method tends more to visit US 

### Age vs gender vs Country 


```python
### check age with countries with gender 
plt.figure(figsize=(30,20))
sns.catplot(x="country_destination",y="age",data=train_df,hue="gender",kind="box",ci=None,col="gender",col_wrap=2,sharey=True,sharex=False)
plt.xlabel("Countries",fontsize="x-large")
plt.ylabel("Age",fontsize="x-large")
##plt.title("Age vs Counties vs gender")
plt.xticks(fontsize="large")
plt.yticks(fontsize="large")
##plt.legend(fontsize="x-large")
plt.savefig("output7.png")

```


    <Figure size 2160x1440 with 0 Axes>



![png](output_41_1.png)


##  Language


```python
## language data 
lan =train_df.groupby(["country_destination","language"]).size()
lan.unstack().plot(kind='bar',figsize=(20,10),stacked=False,width=0.6)
plt.title("Language vs Country",fontsize="xx-large")
plt.xlabel("Country")
plt.xticks(fontsize="x-large")
plt.yticks(fontsize="x-large")
plt.legend(fontsize="large")
plt.savefig("output8.png")
plt.show()
```


![png](output_43_0.png)


after inspecting Language data we find that 98% prefer English as there language

## Age vs Country_destination


```python
### check age with countries
plt.figure(figsize=(30,10))
sns.boxplot(x="country_destination",y="age",data=train_df)
plt.ylabel("Age",fontsize="xx-large")
plt.xlabel("country",fontsize="xx-large")
plt.title("Age vs Countries")
plt.xticks(fontsize="large")
plt.yticks(fontsize="large")
plt.savefig("output10.png")
plt.show()

```


![png](output_46_0.png)


#### after visualizing data we can get to know that older people prefer GB and younger prefer ES

## Signup_method vs Country_destination 


```python
### check signup_method with countries 
plt.figure(figsize=(20,10))
sns.countplot(x="country_destination",data=train_df,hue="signup_method")
plt.xlabel("Countries",fontsize="x-large")
plt.ylabel("signup_method",fontsize="x-large")
plt.title("signup_method vs Counties")
plt.xticks(fontsize="large")
plt.yticks(fontsize="large")
plt.legend(fontsize="x-large")
plt.savefig("output11.png")

```


![png](output_49_0.png)


## Signup_method vs Country_destination  vs Age


```python
### check age with countries with signup_app 
plt.figure(figsize=(30,20))
sns.catplot(x="country_destination",y="age",data=train_df,hue="signup_method",kind="box",ci=None,col="signup_method",col_wrap=3,sharex=False,sharey=True)
plt.xlabel("Countries",fontsize="x-large")
plt.ylabel("Age",fontsize="x-large")
plt.title("Age vs Counties vs signup_method")
plt.xticks(fontsize="large")
plt.yticks(fontsize="large")
plt.legend(fontsize="x-large")
plt.savefig("output12.png")

```


    <Figure size 2160x1440 with 0 Axes>



![png](output_51_1.png)


#### people who use google as there signup_method tends more to book US

## first_device_type vs Country_destination vs Age 


```python
### check age with countries with first_device_type 
plt.figure(figsize=(30,20))
sns.catplot(x="country_destination",data=train_df,hue="first_device_type",kind="count",ci=None,col="gender",col_wrap=2,sharex=False,sharey=False)
plt.xlabel("Countries",fontsize="x-large")
plt.title("Counties vs first_device_type")
plt.xticks(fontsize="large")
plt.yticks(fontsize="large")
##plt.legend(fontsize="x-large")
plt.savefig("output13.png")

```


    <Figure size 2160x1440 with 0 Axes>



![png](output_54_1.png)


#### most users prefer using Mac Desktop


```python
### check age with countries with first_device_type 
plt.figure(figsize=(20,10))
sns.countplot(x="first_device_type",data=train_df)
plt.xlabel("first_device_type",fontsize="x-large")
plt.title("first_device_type")
plt.xticks(fontsize="x-large")
plt.yticks(fontsize="x-large")
plt.show()
plt.savefig("output14.png")
```


![png](output_56_0.png)



    <Figure size 432x288 with 0 Axes>


## Dates 



```python
### dates
train_df['date_account_created'] = pd.to_datetime(train_df['date_account_created'])
train_df['timestamp_first_active'] = pd.to_datetime((train_df.timestamp_first_active)//1000000, format='%Y%m%d')
plt.figure(figsize=(12,6))
train_df.date_account_created.value_counts().plot(kind='line', linewidth=1.2)
plt.xlabel('Date')
plt.title('New account created over time')
plt.show()
```


![png](output_58_0.png)



```python
plt.figure(figsize=(12,6))
train_df.timestamp_first_active.value_counts().plot(kind='line', linewidth=1.2)
plt.xlabel('Date')
plt.title('First Active')
plt.show()
```


![png](output_59_0.png)


#### There is a huge jump in the number of signups and Active member after 2012

# Session 


```python
## session 
session_df=pd.read_csv("airbnb/sessions.csv")   ## Session dataframe
print(session_df.head())
```

          user_id          action action_type        action_detail  \
    0  d1mm9tcy42          lookup         NaN                  NaN   
    1  d1mm9tcy42  search_results       click  view_search_results   
    2  d1mm9tcy42          lookup         NaN                  NaN   
    3  d1mm9tcy42  search_results       click  view_search_results   
    4  d1mm9tcy42          lookup         NaN                  NaN   
    
           device_type  secs_elapsed  
    0  Windows Desktop         319.0  
    1  Windows Desktop       67753.0  
    2  Windows Desktop         301.0  
    3  Windows Desktop       22141.0  
    4  Windows Desktop         435.0  
    

## Null Values


```python
session_df.isnull().sum()
```




    user_id            34496
    action             79626
    action_type      1126204
    action_detail    1126204
    device_type            0
    secs_elapsed      136031
    dtype: int64



### Replacing -unknown- values


```python
## replae all -unknown- values with nan
session_df.replace("-unknown-",np.nan,inplace=True)
```

## Visualizing Device type 
as it's expected the mac device is the most common that users use in there sessions


```python
plt.figure(figsize=(20,10))
sns.countplot(x='device_type', data=session_df)
plt.xlabel('Device type')
plt.ylabel('Number of sessions')
plt.title('Device type distribution')
plt.xticks(rotation=90)
plt.show()
```


![png](output_68_0.png)


# Countries.csv


```python
countries_df=pd.read_csv("airbnb/countries.csv")
all_df=pd.read_csv("airbnb/all.csv")
```


```python
countries_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_destination</th>
      <th>lat_destination</th>
      <th>lng_destination</th>
      <th>distance_km</th>
      <th>destination_km2</th>
      <th>destination_language</th>
      <th>language_levenshtein_distance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AU</td>
      <td>-26.853388</td>
      <td>133.275160</td>
      <td>15297.7440</td>
      <td>7741220.0</td>
      <td>eng</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CA</td>
      <td>62.393303</td>
      <td>-96.818146</td>
      <td>2828.1333</td>
      <td>9984670.0</td>
      <td>eng</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DE</td>
      <td>51.165707</td>
      <td>10.452764</td>
      <td>7879.5680</td>
      <td>357022.0</td>
      <td>deu</td>
      <td>72.61</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ES</td>
      <td>39.896027</td>
      <td>-2.487694</td>
      <td>7730.7240</td>
      <td>505370.0</td>
      <td>spa</td>
      <td>92.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FR</td>
      <td>46.232193</td>
      <td>2.209667</td>
      <td>7682.9450</td>
      <td>643801.0</td>
      <td>fra</td>
      <td>92.06</td>
    </tr>
  </tbody>
</table>
</div>



Getting number of visits per country


```python
list1=[]
for i in countries_df.country_destination : 
    list1.append(sum(train_df["country_destination"] == i ))
list2=[]
for i in countries_df.country_destination :
    for j in range(len(all_df["alpha-2"])): 
        if i == all_df["alpha-2"][j] :
            list2.append(all_df["alpha-3"][j])
```


```python
#### change the value for US since it has most of visits 
list1[-1]=6000
countries_df["visit"]=list1
countries_df["code"]=list2

```


```python
country_geo = 'airbnb/world-countries.json'
data_to_plot = countries_df[['code','visit']]
```


```python
# Setup a folium map at a high-level zoom
map = folium.Map(location=[100, 100], zoom_start=0.5)

# choropleth maps bind Pandas Data Frames and json geometries.
#This allows us to quickly visualize data combinations
map.choropleth(geo_data=country_geo, data=data_to_plot,
             columns=['code','visit'],
             key_on='feature.id',
             fill_color='YlGnBu', fill_opacity=0.7, line_opacity=0.2,
             )
```


```python
map.save('plot_data.html')
```


```python

```

# Age_gender_bkts


```python
age_gender_df=pd.read_csv("airbnb/age_gender_bkts.csv")
```


```python
age_gender_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age_bucket</th>
      <th>country_destination</th>
      <th>gender</th>
      <th>population_in_thousands</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100+</td>
      <td>AU</td>
      <td>male</td>
      <td>1.0</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>95-99</td>
      <td>AU</td>
      <td>male</td>
      <td>9.0</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>90-94</td>
      <td>AU</td>
      <td>male</td>
      <td>47.0</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>85-89</td>
      <td>AU</td>
      <td>male</td>
      <td>118.0</td>
      <td>2015.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>80-84</td>
      <td>AU</td>
      <td>male</td>
      <td>199.0</td>
      <td>2015.0</td>
    </tr>
  </tbody>
</table>
</div>



# Classification Model 
`

loading the datasets again


```python
train_df = pd.read_csv('airbnb/train_users_2.csv')
test_df = pd.read_csv('airbnb/test_users.csv')
df = pd.concat((train_df, test_df), axis=0, ignore_index=True)
## dropping the date_first_booking since it's null in the test dataset
df.drop('date_first_booking', axis=1, inplace=True)
```

## Dealing with dates

### adding more features like (day,month,year,weekday) for both created and first_active


```python
## converting types to datetime type
df['date_account_created'] = pd.to_datetime(df['date_account_created'])
df['timestamp_first_active'] = pd.to_datetime((df.timestamp_first_active // 1000000), format='%Y%m%d')
df['weekday_account_created'] = df.date_account_created.dt.weekday_name
df['day_account_created'] = df.date_account_created.dt.day
df['month_account_created'] = df.date_account_created.dt.month
df['year_account_created'] = df.date_account_created.dt.year
df['weekday_first_active'] = df.timestamp_first_active.dt.weekday_name
df['day_first_active'] = df.timestamp_first_active.dt.day
df['month_first_active'] = df.timestamp_first_active.dt.month
df['year_first_active'] = df.timestamp_first_active.dt.year
```

### Find the time diff between created and first_active 


```python
df['time_lag'] = (df['date_account_created'] - df['timestamp_first_active'])
df['time_lag'] = df['time_lag'].dt.days
df.drop( ['date_account_created', 'timestamp_first_active'], axis=1, inplace=True)
```

after getting the difference between both times then convert it to days (int) then drop data_acount_created and timestamp_first_active since those can't be used in classifier (dates can't be used)

## Age column  

Airbnb rules is that min age for booking is 18 years old 


```python
av=df.age.values
df['age'] = np.where(np.logical_and(av>1915, av<1997), 2015-av, av)
df['age'] = np.where(np.logical_or(av<18, av>100), np.nan, av)
df['age'].fillna(df['age'].mean(),inplace=True)
```

# Session Dataset 


```python
sessions=pd.read_csv("airbnb/sessions.csv")
print(sessions.head())
```

          user_id          action action_type        action_detail  \
    0  d1mm9tcy42          lookup         NaN                  NaN   
    1  d1mm9tcy42  search_results       click  view_search_results   
    2  d1mm9tcy42          lookup         NaN                  NaN   
    3  d1mm9tcy42  search_results       click  view_search_results   
    4  d1mm9tcy42          lookup         NaN                  NaN   
    
           device_type  secs_elapsed  
    0  Windows Desktop         319.0  
    1  Windows Desktop       67753.0  
    2  Windows Desktop         301.0  
    3  Windows Desktop       22141.0  
    4  Windows Desktop         435.0  
    

#### we need to rename the user_id to id to match df 


```python
sessions.rename(columns = {'user_id': 'id'}, inplace=True)
```

insted of multiple rows for each id we will groupby id and each variable and create a new dataframe


```python
action_count = sessions.groupby(['id', 'action'])['secs_elapsed'].agg(len).unstack()
action_type_count = sessions.groupby(['id', 'action_type'])['secs_elapsed'].agg(len).unstack()
action_detail_count = sessions.groupby(['id', 'action_detail'])['secs_elapsed'].agg(len).unstack()
device_type_sum = sessions.groupby(['id', 'device_type'])['secs_elapsed'].agg(sum).unstack()
```


```python
sessions_data = pd.concat([action_count, action_type_count, action_detail_count, device_type_sum],axis=1)
sessions_data.columns=sessions_data.columns.map(lambda x:str(x)+"_count")
sessions_data.index.names = ['id']
```


```python
secs_elapsed = sessions.groupby('id')['secs_elapsed']
secs_elapsed = secs_elapsed.agg(
    {
        'secs_elapsed_sum': np.sum,
        'secs_elapsed_mean': np.mean,
        'secs_elapsed_min': np.min,
        'secs_elapsed_max': np.max,
        'secs_elapsed_median': np.median,
        'secs_elapsed_std': np.std,
        'secs_elapsed_var': np.var,
        'day_pauses': lambda x: (x > 86400).sum(),
        'long_pauses': lambda x: (x > 300000).sum(),
        'short_pauses': lambda x: (x < 3600).sum(),
        'session_length' : np.count_nonzero
    }
)
secs_elapsed.reset_index(inplace=True)
sessions_secs_elapsed = pd.merge(sessions_data, secs_elapsed, on='id', how='left')
df = pd.merge(df, sessions_secs_elapsed, on='id', how = 'left')

```


```python
print(df.shape)
df=df.loc[:,~df.columns.duplicated()]
print(df.shape)
```

    (275547, 571)
    (275547, 529)
    

##  Encoding categorical data



```python
categories = ['gender', 'signup_method', 'signup_flow', 'language','affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser','weekday_account_created', 'weekday_first_active']
df = pd.get_dummies(df, columns=categories)
```

## Splitting back the train and test data


```python
df.set_index('id', inplace=True)
train_df1 = df.loc[train_df['id']]
test_df1 = df.loc[test_df['id']].drop('country_destination', axis=1)
train_df1.reset_index(inplace=True)
test_df1.reset_index(inplace=True)
train_df1.fillna(-1, inplace=True)
test_df1.fillna(-1, inplace=True)

```


```python

```

## Label encoding 


```python
y = train_df1['country_destination']
train_df1.drop(['country_destination','id'], axis=1, inplace=True)
x = train_df1.values
label_encoder = LabelEncoder()
encoded_y_train = label_encoder.fit_transform(y)
```

## Model 


```python
x_train,x_test,y_train,y_test=train_test_split(x,encoded_y_train,test_size=0.2)    
sc_x = StandardScaler() 
x_train = sc_x.fit_transform(x_train)  
x_test = sc_x.transform(x_test)

```


```python
pca=PCA()
x_transform=pca.fit_transform(x_train)
x_test_transform=pca.transform(x_test)
```


```python
models=[RandomForestClassifier(),DecisionTreeClassifier()]
for model in models : 
    print(model)
    model.fit(x_transform,y_train)
    print("Model Trained")
    y_pred=model.predict(x_test)
    print ("Accuracy : ", accuracy_score(y_test, y_pred)) 
    
```

    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
    Model Trained
    Accuracy :  0.5742896629266122
    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')
    Model Trained
    Accuracy :  0.1407322386451477
    


```python
xboost_train=xgb.DMatrix(data=x_transform,label=y_train)
param={"max_depth":2,"seed":42,"learning_rate":1,"n_estimator":5,"objective":"multi:softmax","num_class":12,"nthread":4,"gamma":0,"min_child_weight":1,"colsamole_bytree":1,"colsample_bylevel":1 }
rounds=3
boost=xgb.train(param,xboost_train,rounds)
```


```python
y_pred=boost.predict(xgb.DMatrix(x_test))
```


```python
print ("Accuracy : ", accuracy_score(y_test, y_pred))
```

    Accuracy :  0.5819025087254925
    

#   

# Result  


```python
final=sc_x.transform(test_df1.drop("id",axis=1))
final=pca.transform(final)
y_predict=boost.predict(xgb.DMatrix(final))
results=label_encoder.inverse_transform(y_predict.astype("int32"))
```

    C:\Users\Mohamed\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: DataConversionWarning: Data with input dtype uint8, int64, float64 were all converted to float64 by StandardScaler.
      """Entry point for launching an IPython kernel.
    


```python
submission=pd.DataFrame()
submission["id"]=test_df["id"]
submission["country_destination"]=results
submission.to_csv(r"submission.csv")
```


```python

```


```python

```


```python

```


```python

```
