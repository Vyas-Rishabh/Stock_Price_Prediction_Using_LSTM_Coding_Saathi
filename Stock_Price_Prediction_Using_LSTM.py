#!/usr/bin/env python
# coding: utf-8

# # Stock Price Prediction Using LSTM

# ### Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("Stocks_dataset.csv")
df.head()


# In[3]:


#Shape of Data
df.shape


# In[4]:


#Statistical Descritpion
df.describe()


# In[5]:


#Data Summary
df.info()


# In[6]:


#Check Null values
df.isnull().sum()


# In[7]:


#No null values in data
df = df[['date','open','close']] #Require columns extracting
df['date'] = pd.to_datetime(df['date'].apply(lambda x: x.split()[0])) #Convert to datetime dtype
df.set_index('date', drop=True, inplace=True)
df.head()


# In[8]:


# Now we plotting open and closing price on date index
fig, ax = plt.subplots(1,2, figsize=(20,7))
ax[0].plot(df['open'], label = 'open', color = 'green')
ax[0].set_xlabel('Data', size=13)
ax[0].set_ylabel('Price', size=13)
ax[0].legend()

ax[1].plot(df['close'], label = 'Close', color = 'red')
ax[1].set_xlabel('Data', size=13)
ax[1].set_ylabel('Price', size=13)
ax[1].legend()


# ### Data Preprocessing

# In[9]:


# we'll normalizing all the values of all columns using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
df[df.columns] = mms.fit_transform(df)
df.head()


# In[10]:


#split data into Training and testing
training_size = round(len(df) * 0.75) #75% for training
training_size


# In[11]:


training_data = df[:training_size]
testing_data = df[training_size:]
training_data.shape, testing_data.shape


# In[18]:


# we'll create sequence of data for training and testing

def create_sequence(dataset):
    sequence = []
    labels = []
    
    start_idx = 0
    
    for stop_idx in range(50, len(dataset)): #selecting 50 rows at a time
        sequence.append(dataset.iloc[start_idx:stop_idx])
        labels.append(dataset.iloc[stop_idx])
        start_idx += 1
    return (np.array(sequence), np.array(labels))


# In[19]:


train_seq, train_label, = create_sequence(training_data)
test_seq, test_label = create_sequence(testing_data)
train_seq.shape, train_label.shape, test_seq.shape, test_label.shape


# ### Create LSTM Model

# In[25]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional


# In[27]:


#import Sequential from keras.models
model = Sequential()
#import Dense, Dropout, LSTM, Bidirectional from keras.layers
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_seq.shape[1], train_seq.shape[2])))

model.add(Dropout(0.1))
model.add(LSTM(units=50))

model.add(Dense(2))

model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['mean_absolute_error'])

model.summary()


# In[28]:


# fitting the model by iterating the dataset over 100 times(100 epochs)
model.fit(train_seq, train_label, epochs = 100, validation_data = (test_seq, test_label), verbose = 1)


# In[29]:


# predicting the values after running the model
test_predicted = model.predict(test_seq)
test_predicted[:5]


# In[31]:


# Inversing normalization/scaling on predicted data 
test_inverse_predicted = mms.inverse_transform(test_predicted)
test_inverse_predicted[:5]


# ### PREDICTED DATA VS VISUALIZING ACTUAL

# In[32]:


# Merging actual and predicted data for better visualization
df_merge = pd.concat([df.iloc[-264:].copy(),
                          pd.DataFrame(test_inverse_predicted,columns=['open_predicted','close_predicted'],
                                       index=df.iloc[-264:].index)], axis=1)


# In[34]:


# Inversing normalization/scaling 
df_merge[['open','close']] = mms.inverse_transform(df_merge[['open','close']])
df_merge.head()


# In[35]:


# plotting the actual open and predicted open prices on date index
df_merge[['open','open_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.title('Actual vs Predicted for open price',size=15)
plt.show()


# In[36]:


# plotting the actual close and predicted close prices on date index 
df_merge[['close','close_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.title('Actual vs Predicted for close price',size=15)
plt.show()


# ### PREDICTING UPCOMING 10 DAYS

# In[37]:


# Creating a dataframe and adding 10 days to existing index 

df_merge = df_merge.append(pd.DataFrame(columns=df_merge.columns,
                                        index=pd.date_range(start=df_merge.index[-1], periods=11, freq='D', closed='right')))
df_merge['2021-06-09':'2021-06-16']


# In[38]:


# creating a DataFrame and filling values of open and close column
upcoming_prediction = pd.DataFrame(columns=['open','close'],index=df_merge.index)
upcoming_prediction.index=pd.to_datetime(upcoming_prediction.index)


# In[39]:


curr_seq = test_seq[-1:]

for i in range(-10,0):
  up_pred = model.predict(curr_seq)
  upcoming_prediction.iloc[i] = up_pred
  curr_seq = np.append(curr_seq[0][1:],up_pred,axis=0)
  curr_seq = curr_seq.reshape(test_seq[-1:].shape)


# In[41]:


# inversing Normalization/scaling
upcoming_prediction[['open','close']] = mms.inverse_transform(upcoming_prediction[['open','close']])


# In[42]:


# plotting Upcoming Open price on date index
fig,ax=plt.subplots(figsize=(10,5))
ax.plot(df_merge.loc['2021-04-01':,'open'],label='Current Open Price')
ax.plot(upcoming_prediction.loc['2021-04-01':,'open'],label='Upcoming Open Price')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.set_xlabel('Date',size=15)
ax.set_ylabel('Stock Price',size=15)
ax.set_title('Upcoming Open price prediction',size=15)
ax.legend()
fig.show()


# In[43]:


# plotting Upcoming Close price on date index
fig,ax=plt.subplots(figsize=(10,5))
ax.plot(df_merge.loc['2021-04-01':,'close'],label='Current close Price')
ax.plot(upcoming_prediction.loc['2021-04-01':,'close'],label='Upcoming close Price')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.set_xlabel('Date',size=15)
ax.set_ylabel('Stock Price',size=15)
ax.set_title('Upcoming close price prediction',size=15)
ax.legend()
fig.show()


# You can find the project on <a href="https://github.com/Vyas-Rishabh/Stock_Price_Prediction_Using_LSTM"><b>GitHub.</b></a>
