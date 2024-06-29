import yfinance as yf
import pandas as pd
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class StockNN():
    def __init__(self, ticker, input_num, sma_days, change, save_as_csv=True):
        self.ticker = ticker
        self.input_num = input_num
        self.sma_days = sma_days
        self.change = change
        self.save_as_csv = save_as_csv
        self.data = self.__fetch_stock_data()
        self.cols, self.X, self.y = self.__process_data()

        
    def __fetch_stock_data(self):
        start_date = pd.Timestamp.today() - pd.DateOffset(years=10)
        end_date = pd.Timestamp.today()
        data = yf.download(self.ticker, start=start_date, end=end_date)


        data['Change'] = data['Close'] - data['Open']
        data['Change%'] = round(((data['Close'] - data['Open'])/data['Open']+0.5),4)
        data['Fluctuate'] = data['High'] - data['Low']
        data['Fluctuate%'] = round(((data['High'] - data['Low'])/data['Low']),4)
        data['Body'] = round((abs(data['Change'])/data['Fluctuate']),4)
        data['Head_Distance%'] = data.apply(lambda row: round((row['High'] - row['Close'])/row['Fluctuate'], 4)\
                                            if row['Change'] > 0 \
                                            else round((row['High'] - row['Open'])/row['Fluctuate'], 4), axis=1)
        data['Leg_Distance%'] = data.apply(lambda row: round((row['Open'] - row['Low'])/row['Fluctuate'], 4)\
                                            if row['Change'] > 0 \
                                            else round((row['Close'] - row['Low'])/row['Fluctuate'], 4), axis=1)
        
        # Calculate for SMAs
        for i in range(25,200, 25):
            data[f'{i}sma'] = data['Close'].rolling(window=i).mean()
            data[f'Distance_{i}'] = round(((data['Close'] - data[f'{i}sma'])/data['Close']+0.5),2)

        if (self.save_as_csv):
            # Drop rows with missing values
            data.dropna(inplace=True)
            directory = 'Analysis/Stock data'
            # Save data as CSV file
            if not os.path.exists(directory):
                os.makedirs(directory)
            data.to_csv(f"{directory}/{self.ticker}.csv")

        return data
    
    def __process_data(self):    
        cols , X, y = self.__method_2()
        print("len of data: ", len(self.data))
        print("shape of X: ", X.shape) 
        print("shape of y ", y.shape)

        self.__normalize_data(X)
        # self.__standardize_data(X)
        return cols, X, y

    def __method_1(self, label_avg_num, sma_day):
        input_params = ['Change%', 'Fluctuate%', 'Body', 'Head_Distance%', 'Leg_Distance%', f'Distance_{sma_day}']
        for j in range(len(self.data)-self.input_size-label_avg_num+1):
            # input data handling
            sample = []
            for i in range(self.input_size):
                buf=[]
                for param in input_params:
                    buf.append(self.data[param].iloc[j+i])
                sample.append(buf)
            self.X.append(sample)

            # label data handling
            moving_avg = self.data[f'Distance_{sma_day}'].rolling(window=2).mean().iloc[self.input_size+j:self.input_size+j+label_avg_num]
            if moving_avg.iloc[-1] > moving_avg.iloc[0]:
                self.y.append([1,0,0])  # Up trend
            elif moving_avg.iloc[-1] < moving_avg.iloc[0]:
                self.y.append([0,0,1])  # Down trend
            else:
                self.y.append([0,1,0])
            
        return len(input_params), np.array(self.X), np.array(self.y)

    def __method_2(self):    # find the pattern before big change
        if(self.change < 0):
            print("Change value should be positive")
            return -1
        if(self.sma_days not in [25,50,75,100,125,150,175,200]):
            print("SMA day should be one of the following: 25,50,75,100,125,150,175,200")
            return -1
        
        input_params = [ 'Body', 'Head_Distance%', 'Leg_Distance%', f'Distance_{self.sma_days}']
        X, y = [],[]

        for j in range(self.input_num,len(self.data)):
            # input data handling
            if(self.data['Change%'].iloc[j] > self.change or self.data['Change%'].iloc[j] < -self.change):
                sample = []
                for i in range(self.input_num):
                    buf=[]
                    for param in input_params:
                        buf.append(self.data[param].iloc[j-i])
                    sample.append(buf)    
                sample = sample[::-1]         
                X.append(sample)

                if(self.data['Change%'].iloc[j] >= self.change):
                    y.append(1)
                elif(self.data['Change%'].iloc[j] < -self.change):
                    y.append(0)

        return len(input_params), np.array(X), np.array(y)
    
    def __standardize_data(self, X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)
    
    def __normalize_data(self, X):
        return X / np.linalg.norm(X)

    def __create_neural_network(self):
        try:
            # Define the model architecture
            model = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(shape=(self.input_num, self.cols)),
                tf.keras.layers.Flatten(),  # Correct input shape for Flatten layer
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')  
            ])

            # Compile the model with an appropriate loss function
            model.compile(optimizer='adam',
                            loss='binary_crossentropy',  
                            metrics=['accuracy'])
            return model
        except:
            print("Error in creating the model")
            return -1
    

    def run(self, display=False):
        # Create the neural network
        model = self.__create_neural_network()
        if(model == -1):
            print("Error in creating the model")
            return
        if(display):
            print('X: ', self.X)
            print('y: ', self.y)
            print('model summary: ')
            model.summary()

        # Split X and y for training dataset
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        print("X_train shape: ", X_train.shape)
        print("y_train shape: ", y_train.shape)
        print("X_test shape: ", X_test.shape)
        print("y_test shape: ", y_test.shape)

        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

        # Train the neural network
        model.fit(X_train, y_train, epochs=10, batch_size=2, validation_split=0.2, callbacks=[early_stopping])

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
        print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

def main():
    stockNN = StockNN('TSLA', input_num=10, sma_days=75, change=0.05)
    stockNN.run(display=True)


if __name__ == "__main__":
    main()