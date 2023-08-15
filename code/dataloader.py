#%% 
import numpy as np
import pandas as pd

import os

# source: https://lobsterdata.com/info/DataSamples.php



# %%
# traverse file tree and return all files in a list
def get_file_list(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

# %%

path = './data'

file_list = get_file_list(path)

# %%

# read in all csv files, ignore ones not in csv format
df_dict = {} # format: 'filename': df
for file in file_list:
    try:
        df_dict[file] = pd.read_csv(file)
    except:
        pass

# %%

# strip the keys of the dictionary to just the stock name and orderbook/message
# e.g. './data/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv' -> 'AMZN_orderbook'
df_dict = {k.split('/')[-1].split('_')[0] + '_' + k.split('/')[-1].split('_')[-2]: v for k, v in df_dict.items()}
# %%

# Set column names for 'message' files
message_cols = ['Time', 'Type', 'Order_ID', 'Size', 'Price', 'Direction']

# Set base column names for 'orderbook' files
orderbook_base_cols = ['Ask_Price', 'Ask_Size', 'Bid_Price', 'Bid_Size']

for key, df in df_dict.items():
    if 'message' in key:
        df.columns = message_cols
    elif 'orderbook' in key:
        # Determine the number of levels in the orderbook file
        levels = df.shape[1] // 4
        # Generate column names for all levels
        orderbook_cols = [f'{col}_{level}' for level in range(1, levels + 1) for col in orderbook_base_cols]
        df.columns = orderbook_cols

# %%
# Create a dictionary to store the merged dataframes
merged_dfs = {}

# Get unique stocks
stocks = set([key.split('_')[0] for key in df_dict.keys()])

for stock in stocks:
    # Get the corresponding message and orderbook dataframes
    message_df = df_dict.get(f'{stock}_message')
    orderbook_df = df_dict.get(f'{stock}_orderbook')

    if message_df is not None and orderbook_df is not None:
        # Merge the dataframes based on the index
        merged_df = pd.concat([message_df, orderbook_df], axis=1)

        # Store the merged dataframe in the dictionary
        merged_dfs[f'{stock}_df'] = merged_df

# Now, merged_dfs dictionary contains the merged dataframes for each stock,
# for example, to access the dataframe for 'msft', you would use: merged_dfs['msft_df']

# add additional time column formatted in HH:MM:SS:MS

#%%
def convert_time(t):
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    milliseconds = int((t - int(t)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03}"


for stock in stocks:
    df = merged_dfs[f'{stock}_df']
    df['time_adjusted'] = df['Time'].apply(convert_time)
    df['datetime'] = pd.to_datetime(df['Time'], unit='s', origin=pd.Timestamp('2012-06-21'))
    # make 'datetime' the first column
    datetime_col = df.pop('datetime')
    df.insert(0, 'datetime', datetime_col)



msft_df = merged_dfs['MSFT_df']
#%%
# msft_df_condensed = msft_df.copy()
# msft_df_condensed = msft_df_condensed.set_index('datetime')
# msft_df_condensed = msft_df_condensed.resample('250ms').last()
# msft_df_condensed = msft_df_condensed.reset_index()

resampled_dfs = {}
for stock in stocks:
    # print('resampling', stock)
    df = merged_dfs[f'{stock}_df'].copy()
    df = df.set_index('datetime')
    df = df.resample('250ms').last()
    df = df.reset_index()
    df = df.fillna(method='ffill')
    resampled_dfs[f'{stock}_df'] = df
    print(f'{stock} shape: ', df.shape)

#%%
# export to csv 
for stock in stocks:
    df = resampled_dfs[f'{stock}_df']
    df.to_csv(f'./data/dataframes/{stock}_df_resampled.csv')
    




# print index of first elapsed second in each df



#%%

merged_dfs['MSFT_df'].to_csv('./data/dataframes/MSFT_df.csv')
merged_dfs['INTC_df'].to_csv('./data/dataframes/INTC_df.csv')
merged_dfs['AAPL_df'].to_csv('./data/dataframes/AAPL_df.csv')
merged_dfs['AMZN_df'].to_csv('./data/dataframes/AMZN_df.csv')
merged_dfs['GOOG_df'].to_csv('./data/dataframes/GOOG_df.csv')

# %%

# plot the four stocks on the same graph

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(20, 10))

for stock in stocks:
    df = merged_dfs[f'{stock}_df']
    ax[0, 0].plot(df['Time'], df['Price'], label=stock)
    ax[0, 0].set_title('Price vs Time')
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Price')
    ax[0, 0].legend()

    ax[0, 1].plot(df['Time'], df['Size'], label=stock)
    ax[0, 1].set_title('Size vs Time')
    ax[0, 1].set_xlabel('Time')
    ax[0, 1].set_ylabel('Size')
    ax[0, 1].legend()

    ax[1, 0].plot(df['Time'], df['Ask_Price_1'], label=stock)
    ax[1, 0].set_title('Ask Price vs Time')
    ax[1, 0].set_xlabel('Time')
    ax[1, 0].set_ylabel('Ask Price')
    ax[1, 0].legend()

    ax[1, 1].plot(df['Time'], df['Bid_Price_1'], label=stock)
    ax[1, 1].set_title('Bid Price vs Time')
    ax[1, 1].set_xlabel('Time')
    ax[1, 1].set_ylabel('Bid Price')
    ax[1, 1].legend()

plt.show()
# %%

# plot four prices in one graph
fig, ax = plt.subplots(4, 1, figsize=(20, 15))
idx = 0
for stock in stocks:
    df = merged_dfs[f'{stock}_df']
    ax[idx].plot(df['Time'], df['Bid_Price_1'], label=stock)
    ax[idx].set_title('Price vs Time')
    ax[idx].set_xlabel('Time')
    ax[idx].set_ylabel('Price')
    ax[idx].legend()
    idx += 1



# %%



