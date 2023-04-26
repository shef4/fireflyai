#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import json
import sklearn
from sklearn.model_selection import train_test_split
import joblib

#imports for random forest
from sklearn.ensemble import RandomForestClassifier

# imports for KNN
from sklearn.neighbors import KNeighborsClassifier

#imports for LSTM
from numpy import zeros, newaxis
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential 
from keras import regularizers
from keras import Input
from keras.layers import Bidirectional, LSTM, RepeatVector, Dense, TimeDistributed, Dropout
from pandas.core.describe import describe_timestamp_as_categorical_1d

import tkinter
from tkinter import ttk
import random


class Data_Processor():
    def __init__(self):
        self.user_types = ["Worker", "Student", "Senior Citizen"]
        self.noise_types = ["Small", "Medium", "Large"]
        self.has_pet = [True, False]
        self.has_plant = [True, False]
        self.sleep_temp = [x for x in range(60, 68)]
        self.noise_value = { "Small": random.uniform(-1, 1), "Medium": random.uniform(-1, 2), "Large": random.uniform(-2, 3)}
        self.days_of_week = { "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        self.seasonal_temp = { 1: 68, 2: 75, 3: 78, 4: 72}# 1-winter 2-spring 3-summer 4-fall
        
    # convert to celsius
    def ftc(self, temp):
        return (temp - 32) * 5 / 9

    # convert to fahrenheit
    def ctf(self, temp):
        return temp * 9 / 5 + 32

    def format_data(self, data_df):
        data_df.columns = ['WBANNO', 'UTC_DATE', 'UTC_TIME', 'LST_DATE', 'LST_TIME', 'CRX_VN', 'LONGITUDE', 'LATITUDE', 'AIR_TEMPERATURE',
                        'PRECIPITATION', 'SOLAR_RADIATION', 'SR_FLAG', 'SURFACE_TEMPERATURE', 'ST_TYPE', 'ST_FLAG', 'RELATIVE_HUMIDITY',
                        'RH_FLAG', 'SOIL_MOISTURE_5', 'SOIL_TEMPERATURE_5', 'WETNESS', 'WET_FLAG', 'WIND_1_5', 'WIND_FLAG']
        # drop useless columns
        data_df.drop(['WBANNO', 'LST_DATE', 'LST_TIME', 'CRX_VN', 'PRECIPITATION', 'SR_FLAG', 'SOLAR_RADIATION', 'SURFACE_TEMPERATURE',
                    'SOIL_MOISTURE_5', 'SOIL_TEMPERATURE_5', 'LONGITUDE', 'LATITUDE', 'ST_TYPE', 'ST_FLAG', 'RH_FLAG', 'WETNESS',
                    'WET_FLAG', 'WIND_1_5', 'WIND_FLAG'], axis = 1, inplace = True)
        # rename columns
        data_df.rename(columns ={ "AIR_TEMPERATURE": "EXTERNAL_TEMP"}, inplace = True)
        data_df.rename(columns ={ "RELATIVE_HUMIDITY": "OUTSIDE_HUMIDITY"}, inplace = True)
        data_df.rename(columns ={ "UTC_DATE": "DATE"}, inplace = True)
        data_df.rename(columns ={ "UTC_TIME": "TIME"}, inplace = True)
        # format date and time
        unformated_date_time = data_df[['DATE', 'TIME']].to_numpy()
        data_df['DATE'] = data_df['DATE'].apply(lambda x: '{0:0>8}'.format(x))
        data_df['DATE'] = pd.to_datetime(data_df['DATE'], format = '%Y%m%d')
        data_df['TIME'] = data_df['TIME'].apply(lambda x: '{0:0>4}'.format(x))
        data_df['TIME'] = pd.to_datetime(data_df['TIME'], format = '%H%M').dt.time
        return data_df.to_numpy(), unformated_date_time

    def save_data(self, data, user_profile):
        if not os.path.exists('data'):
            os.makedirs('data')
        np.save(f'data/processed_data.npy', data)
        # save user profile as txt
        with open(f'data/user_profile_{user_profile["user_type"]}.txt', 'w') as f:
            f.write(json.dumps(user_profile))
        return

    # generate random day events
    def gen_schedule(self, first_event, num_events, noise_type):
        for i in range(num_events + 2):
            end_value = 0
            if i == 0:  # first block is sleep
                start, end = -1, 6
            elif i == 1:  # next blocks are work
                start, end = max(first_event[0], end_value), first_event[1]
            elif i == num_events+1:  # last block is sleep
                start, end = max(22, end_value), float('inf')
            # random intervals within bounds
            noise = self.noise_value[noise_type]
            start_value = int(min(23, max(random.uniform(start - noise, start + noise), 0)))
            end_value = int(min(23, max(random.uniform(end - noise, end + noise), 0)))
            # catch errors
            if start_value == end_value and start_value == 23:
                start_value, end_value = start_value - 1, end_value
            elif start_value == end_value:
                start_value, end_value = start_value, end_value + 1
            start = end + 2
            end = start + random.randint(2, 3)
            yield(start_value, end_value)

    # generate random profile
    def gen_user_profile(self, user_type= None, noise_type= None, pets= None, plants= None, insolation_time_ac= None, insolation_time_external= None, schedule_cycle= None):
        profile = {"user_type": user_type if user_type else random.choice(self.user_types),
                        "noise_type": noise_type if noise_type else random.choice(self.noise_types),
                        "pets": pets if pets else random.choice(self.has_pet),
                        "plants": plants if plants else random.choice(self.has_plant),
                        "sleep_temp": random.choice(self.sleep_temp),
                        "insolation_time": {"ac_temp": insolation_time_ac if insolation_time_ac else random.uniform(1, 0.1),
                                            "external_temp": insolation_time_external if insolation_time_external else random.uniform(1, 0.1)}}
        num_days = schedule_cycle if schedule_cycle else 7
        # monday=0, tuesday=1, wednesday=2, thursday=3, friday=4, saturday=5, sunday=6
        schedule = []
        for day in range(num_days):
                    first_event = (7, 10)
                    num_events = random.randint(1, 4)
                    gen_day = list(self.gen_schedule((7, 16), 1, profile["noise_type"])) if profile["user_type"] == "Worker" and day % 7 < 5 else list(self.gen_schedule(first_event, num_events, profile["noise_type"]))
                    if profile["user_type"] == "Student":
                        if day % 7 in (5, 6):
                            day_schedule = gen_day
                        elif day % 7 in (1, 3):
                            day_schedule = gen_day if day % 7 == 1 else copy_tuesday
                            copy_tuesday = day_schedule
                        elif day % 7 in (0, 2, 4):
                            day_schedule = gen_day if day % 7 == 0 else copy_monday
                            copy_monday = day_schedule
                    else:
                        day_schedule = gen_day
                    schedule.append(day_schedule)
        profile["schedule"] = schedule
        return profile

    def process_data(self, filename, user_profile, quick=False):
        # process data Date=0, Time=1, Outside Temp=2, Outside Humidity=3, AC_TEMP=4, INTERNAL_TEMP=5, AI_CHANGE=6, USER_CHANGE=7, DAY_OF_WEEK=8, SEASON=9
        raw_data_df = pd.read_csv(filename+'.txt', sep='\s+', header=None)
        # convert raw_data to numpy array data
        data, unformated_date_time = self.format_data(raw_data_df)
        # add new columns on end
        data = np.hstack((data, np.full((len(data), 6), None)))
        # initializes variables
        sleep_temp = self.ftc(user_profile["sleep_temp"])
        pet_temp = random.uniform(self.ftc(64), self.ftc(78))
        plant_temp = random.uniform(self.ftc(60), self.ftc(75))
        season_temp = self.ftc(self.seasonal_temp[data[0][0].month % 12 // 3 + 1] + self.noise_value[user_profile["noise_type"]])
        ac_delay = user_profile["insolation_time"]["ac_temp"] * 2 + 0.5*self.noise_value[user_profile["noise_type"]]
        schedule_cycle = len(user_profile["schedule"])
        schedule_idx = self.days_of_week[data[0, 0].day_name()]
        day_value = 288
        hour_value = 12
        num_days = len(data) // day_value
        for day_idx in range(num_days):
            today = day_idx * day_value
            tomorrow = today + day_value
            day_of_week = self.days_of_week[data[today, 0].day_name()]
            data[today:tomorrow, 4:10] = [-99, -99, 0, 0, day_of_week % 7, ((data[today, 0].month % 12) // 3) + 1]
            if today > 0:
                yesterday = today - day_value
                if data[today, 9] != data[yesterday, 9]:  # new season
                    season_temp = self.ftc(self.seasonal_temp[data[today, 9]] + self.noise_value[user_profile["noise_type"]])
                    plant_temp += self.noise_value[user_profile["noise_type"]]
                if data[today, 0].month != data[yesterday, 0].month:  # new month
                    sleep_temp += self.noise_value[user_profile["noise_type"]]
                    if data[today, 0].month % 3 == 0:
                        pet_temp += self.noise_value[user_profile["noise_type"]]
            schedule = user_profile["schedule"][schedule_idx % schedule_cycle]
            schedule_idx += 1
            for event_idx in range(len(schedule)):
                start_time, end_time = schedule[event_idx]
                sleeping = (start_time == 0) or (end_time == 23)
                start_idx = today + start_time * hour_value
                end_time = end_time+1 if end_time == 23 else int(end_time - ac_delay+1)
                end_idx = min(start_idx + (end_time - start_time) * hour_value, len(data))
                if sleeping:
                    data[start_idx:end_idx, 4] = sleep_temp
                    data[start_idx, 7] = 1 if end_time == 23 else 0
                else:
                    if user_profile["pets"] and user_profile["plants"]:
                        data[start_idx:end_idx, 4] = (pet_temp + plant_temp)/2
                    elif user_profile["pets"] or user_profile["plants"]:
                        data[start_idx:end_idx, 4] = plant_temp if user_profile["plants"] else pet_temp
                    else:
                        data[start_idx:end_idx, 4] = -99
                    data[start_idx, 7] = 1   
                if event_idx < len(schedule)-1:
                    start_idx = end_idx
                    end_idx = min(start_idx + (schedule[event_idx+1][0]-end_time)*hour_value, len(data))
                    data[start_idx:end_idx, 4] = season_temp
                    data[start_idx, 7] = 1         
        # set internal temp
        for row in range(len(data)):
            col, factor = (2, "external_temp") if data[row, 4] == -99 else (4, "ac_temp")
            if row == 0:
                data[row, 5] = data[row, col]
            else:
                delta = (data[row, col] - data[row-1, 5]) / (user_profile["insolation_time"][factor]*60)
                data[row, 5] = data[row-1, 5] + delta      
        # concat raw date time
        data = np.hstack((unformated_date_time[:, :], data[:, [2, 3, 4, 5, 6, 7, 8, 9]]))
        # save data
        if not quick:
            self.save_data(data, user_profile)
        return data
    
    # plot data
    def plot_data(self, processed_data_df, user_profile, show_ac_temp=True, show_inside_temp=True, show_outside_temp=True, convert=True, time_type="day"):
        print("user_type: ", user_profile ["user_type"])
        print("noise_type: ", user_profile ["noise_type"])
        print("pets: ", user_profile ["pets"])
        print("plants: ", user_profile ["plants"])
        print("insolation_time: ", user_profile ["insolation_time"])
        x_axis_label = { "day": 288, "week": 7 * 288, "month": 288 * 7 * 4, "year": 288 * 52 * 7}
        #if ac temp is -99, replace 0
        processed_data_df = processed_data_df.replace(-99, 0)
        # filter outlines >-20
        processed_data_df = processed_data_df[processed_data_df['AC Temp'] > -20]
        processed_data_df = processed_data_df[processed_data_df['Internal Temp'] > -20]
        processed_data_df = processed_data_df[processed_data_df['External Temp'] > -20]
        # plot data
        y = None
        plt.figure(figsize=(20, 10))
        if show_ac_temp:
            if convert:
                y = self.ctf(processed_data_df['AC Temp'][:x_axis_label[time_type]])
            else:
                y = processed_data_df['AC Temp'][:x_axis_label[time_type]]
            plt.plot(y, label="AC Temp", color="red")
        if show_inside_temp:
            if convert:
                y = self.ctf(processed_data_df['Internal Temp']
                        [:x_axis_label[time_type]])
            else:
                y = processed_data_df['Internal Temp'][:x_axis_label[time_type]]
            plt.plot(y, label="Internal Temp", color="blue")
        if show_outside_temp:
            if convert:
                y = self.ctf(processed_data_df['External Temp'][:x_axis_label[time_type]])
            else:
                y = processed_data_df['External Temp'][:x_axis_label[time_type]]
            plt.plot(y, label="External Temp", color="green")
        plt.legend()
        plt.show()
        
  
class Model_LSTM():
    def __init__(self):
# create default model
        self.model = None
        self.history = None
        self.mean = None
        self.std = None

    def format_data(self, data):
        timesteps = 6
        shift = 288*7  
        # drop date time column
        data = data[:,1:]
        #data = data[:len(data) - len(data) % timesteps+288]
        
        df = pd.DataFrame(data, columns=['time','External Temp', 'Outside Humidity', 'AC Temp', 'Internal Temp', 'AI Change', 'User Change', 'Day of Week', 'Season'])

        ######## split time into hourly interval and ohc
        df['time'] = df['time']/100
        swap_list = ['External Temp', 'Outside Humidity', 'AC Temp', 'Internal Temp', 'AI Change', 'User Change', 'Day of Week', 'Season','time']
        df = df.reindex(columns=swap_list)

        #one hot coding
        #df['time'] = pd.Categorical(df['time'].astype(str), categories=['0', '1', '2', '3', '4', '5', '6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'])
        #df = pd.get_dummies(df, columns=['time'])
        df['Day of Week'] = pd.Categorical(df['Day of Week'].astype(str), categories=['0', '1', '2', '3', '4', '5', '6'])
        df = pd.get_dummies(df, columns=['Day of Week'])
        df['Season'] = pd.Categorical(df['Season'].astype(str), categories=['1', '2', '3', '4'])
        df = pd.get_dummies(df, columns=['Season'])
        # Normalize numeric data
        norm_col = ['External Temp', 'Outside Humidity', 'AC Temp', 'Internal Temp','time']
        self.mean = df['AC Temp'].mean()
        self.std = df['AC Temp'].std()
        df[norm_col] = (df[norm_col] - df[norm_col].mean()) / df[norm_col].std()
        df.drop(['Internal Temp'], axis=1, inplace=True)

        data = df.to_numpy()
        x_data_df = df.copy()
        x_data_df.drop(['AC Temp'], axis=1, inplace=True)
        x_data = x_data_df.to_numpy()
        x_data = x_data[:x_data.shape[0]-shift]
        y_data = df['AC Temp'].to_numpy()
        y_data = y_data[shift:y_data.shape[0]]
        x_data = np.reshape(x_data, (int(x_data.shape[0]/timesteps), timesteps, x_data.shape[1]))
        y_data = np.reshape(y_data, (y_data.shape[0], 1, 1))
        y_data = np.reshape(y_data, (int(y_data.shape[0]/timesteps), timesteps, 1))
        return x_data.astype('float32'), y_data.astype('float32')

    def train(self, data, batch_size=24, epochs=100):
        ######################
        #total data: 40 days
        #training: 28 days
        #testing: 12 days
        ######################
        data = data[:11520]
        train, val = train_test_split(data, test_size=0.3, shuffle=False)
        #train = data[:8065]
        #val = data[8065:11522]
        print(train.shape)
        print(val.shape)
        x_train, y_train = self.format_data(train)
        x_val, y_val = self.format_data(val)
        print(x_train.shape)
        print(x_val.shape)

        ############ changed model into stateful model
        ############ note that number of test/train/validation data entries must be divisible by batchsize(24)
        # train model
        self.model = Sequential(name="LSTM-Model")  # Model
        self.model.add(LSTM(64, batch_input_shape=(batch_size,x_train.shape[1], x_train.shape[2]), return_sequences=True, stateful=True))
        #self.model.add(LSTM(128, return_sequences=True))
        #self.model.add(LSTM(64, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1, activation='linear'))  # Output Layer
        self.model.summary()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3, mode='min')
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), metrics=[tf.keras.metrics.MeanAbsoluteError()])
        self.history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,callbacks=[early_stopping], validation_data=(x_val, y_val), verbose=2, shuffle=False)
        # cross validate models acc
        acc = self.history.history['val_loss'][-1]
        return acc

    def test(self, data):
        # test models acc
        x_test, y_test = self.format_data(data)
        #acc = self.model.evaluate(x_test, y_test, batch_size=16)[1]
        acc = 0
        return acc
      
    def predict(self, data):
        # give model prediction and actual value
        inputs, labels = self.format_data(data)
        #inputs = inputs[0:1728,:]
        #labels = labels[0:1728,:]
        labels = labels*self.std+self.mean
        predictions = self.model.predict(inputs,batch_size=24)*self.std+self.mean
        return predictions, labels

    def load(self, filename):
        #load model
        self.model = joblib.load(filename)

    def save(self, filename):
        # save model
        if not os.path.exists('models/lstm'):
            os.makedirs('models/lstm')
        joblib.dump(self.model, 'models/lstm'+filename)
        
if __name__ == "__main__":
    LSTM_model= Model_LSTM()
    Processer = Data_Processor()
    user_profile = Processer.gen_user_profile(user_type= "Senior Citizen")
    Processed_data=Processer.process_data("raw_data", user_profile)
    LSTM_model.train(data=Processed_data)

    tempIn = 100 # call variable from the other file and display it here
    '''
    def increaseTemp():
    # increase temperature by 1 degree
        tempIn = tempIn.get() 
        tempIn = tempIn + 1
    '''
        
    window = tkinter.Tk() # parent window - largest box
    window.title("ThermoFli Dashboard")


    # create a frame
    frame = tkinter.Frame(window)
    frame.pack()

    # user info frame
    user_info_frame = tkinter.LabelFrame(frame, text="User Info")
    user_info_frame.grid(row=0, column=0, padx=20, pady=10)

    # taking user information

    # are there plants in the house 
    plants_lable= tkinter.Label(user_info_frame, text="Plants:")
    plants_combobox = ttk.Combobox(user_info_frame, values=["Yes", "No"])
    plants_lable.grid(row=0, column=1)
    plants_combobox.grid(row=1, column=1)

    # are there plants in the house 
    pets_lable= tkinter.Label(user_info_frame, text="Pets:")
    pets_combobox = ttk.Combobox(user_info_frame, values=["Yes", "No"])
    pets_lable.grid(row=0, column=0)
    pets_combobox.grid(row=1, column=0)

    # User type input box and label
    title_lable= tkinter.Label(user_info_frame, text="User Type:")
    title_combobox = ttk.Combobox(user_info_frame, values=["Student", "Worker", "Senior Citizen"])
    title_lable.grid(row=0, column=2)
    title_combobox.grid(row=1, column=2)

    # User type input box and label
    noise_lable= tkinter.Label(user_info_frame, text="Noise Type:")
    noise_combobox = ttk.Combobox(user_info_frame, values=["Small", "Medium", "large"])
    noise_lable.grid(row=0, column=3)
    noise_combobox.grid(row=1, column=3)


    # setting the padding for the widgets
    for widget in user_info_frame.winfo_children():
        widget.grid_configure(padx=15, pady=10)
        
    # saving the information
    main_settings = tkinter.LabelFrame(frame, text="Temperature Settings")
    main_settings.grid(row=1, column=0, sticky="news", padx=20, pady=20)

    # displaying the graph
    def plot_prediction(user_profile, data, time_range = "week"):
        time = {"day": 288, "week": 7*288, "month": 288*7*4, "year": 288*52*7}
        
        predictions, labels = LSTM_model.predict(data)
       
        processed_data_df = pd.DataFrame(data,columns=['Date','time','External Temp', 'Outside Humidity', 'AC Temp', 'Internal Temp', 'AI Change', 'User Change', 'Day of Week', 'Season'])# add user info on plot
        
        print("user_type: ", user_profile["user_type"])
        print("noise_type: ", user_profile["noise_type"])
        print("pets: ", user_profile["pets"])
        print("plants: ", user_profile["plants"])
        print("insolation_time: ", user_profile["insolation_time"])

        # if ac temp is -99, replace 0
        processed_data_df = processed_data_df.replace(-99, 0)
        # filter outlines >-20
        processed_data_df = processed_data_df[processed_data_df['AC Temp'] > -20]
        processed_data_df = processed_data_df[processed_data_df['Internal Temp'] > -20]
        processed_data_df = processed_data_df[processed_data_df['External Temp'] > -20]
        # plot data
        #user_info = f"user_type: {user_profile['user_type']}\nnoise_type: {user_profile['noise_type']}\npets: {user_profile['pets']}\nplants: {user_profile['plants']}\ninsolation_time: {user_profile['insolation_time']}"
        plt.plot(predictions.flatten()[:time[time_range]])
        plt.plot(labels.flatten()[:time[time_range]])
        plt.plot(processed_data_df['Internal Temp'][:time[time_range]], label="Internal Temp", color="blue")
        plt.plot(processed_data_df['External Temp'][:time[time_range]], label="External Temp", color="green")
        plt.title('model prediction VS actual')
        plt.legend(['prediction', 'actual', 'Internal Temp', 'External Temp'], loc='upper left')
        #plt.text(0.5, -0.2, user_info, ha='center', va='top')
        plt.show()

    def graph():
        processor = Data_Processor()
        plants = plants_combobox.get()
        pets = pets_combobox.get()
        occupation = title_combobox.get()
        sound = noise_combobox.get()
        time_range="week"
        insolation_time_ac = random.uniform(1,0.2)
        insolation_time_external = random.uniform(1,0.2)
        schedule_cycle = 7
        user_profile = processor.gen_user_profile(occupation, sound, pets, plants, insolation_time_ac, insolation_time_external, schedule_cycle)
        data = processor.process_data("raw_data", user_profile, quick = True)
        plot_prediction(user_profile, data, time_range)

    mybuttonn = tkinter.Button(window, text="Graph", command=graph, width=10, height=2)
    mybuttonn.pack()


    window.mainloop() 
