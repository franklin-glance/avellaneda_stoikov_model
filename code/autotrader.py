#%% 
import collections
import itertools
import math

import numpy as np
import pandas as pd
import scipy

from typing import List

from env import * 
import asyncio
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'

MINIMUM_BID = 0
MAXIMUM_ASK = 9999999999999


LOT_SIZE = 10
POSITION_LIMIT = 100 
TICK_SIZE_IN_CENTS = 100
MIN_BID_NEAREST_TICK = (MINIMUM_BID + TICK_SIZE_IN_CENTS) // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS
MAX_ASK_NEAREST_TICK = MAXIMUM_ASK // TICK_SIZE_IN_CENTS * TICK_SIZE_IN_CENTS


TICK_SIZE_IN_CENTS = 100
PHI_MAX = 10 # max position
PHI_ASK = 10
PHI_BID = 10
GAMMA = 2 
ORDER_DELAY = 1 # wait time before cancelling active unfilled order

USE_TRADE_TICKS = False# update market on trade ticks as well

KAPPA_SAMPLING_TIME = 20  # seconds of history looked at to determine trading intensity (default = 20) 
KAPPA_SAMPLING_TICK_RANGE = 4 # number of ticks to sample for kappa (up and down) (Default = 4) 
MIN_ORDER_INTENSITY_SAMPLES = 10 
INTENSITY_WINDOW = 10   # window for recording order intensity

HEDGE_SCALE_FACTOR = 0 # 1 to fully hedge

T = 1 # CONSTANT FOR INFINITE TIME WINDOW 

USE_VAMP = True # if true, use vamp as price, false, use midprice

def round_price(price) -> int:
    """returns price rounded to tick size"""
    return round(price / TICK_SIZE_IN_CENTS) * TICK_SIZE_IN_CENTS


class OrderBook:
    def __init__(self, bid_prices: List[int], bid_volumes: List[int], ask_prices: List[int], ask_volumes: List[int], timestamp) -> None:
        self.timestamp = timestamp
        self.bid_prices = bid_prices
        self.bid_volumes = bid_volumes
        self.ask_prices = ask_prices
        self.ask_volumes = ask_volumes

        self.total_vol = sum(ask_volumes[i] + bid_volumes[i] for i in range(len(ask_volumes)))

        self.best_bid = self.bid_prices[0]
        self.best_ask = self.ask_prices[0]
        
        self.mid_price = (self.best_ask + self.best_bid) // 2 # midpoint price 
        # 
        self.imbalance = (self.bid_volumes[0]) / (self.bid_volumes[0] + self.ask_volumes[0])
        self.weighted_mid_price = (self.imbalance)*self.ask_prices[0] + (1-self.imbalance)*self.bid_prices[0] 

        self.bid_price_vwap = sum([self.bid_prices[i]*self.bid_volumes[i] for i in range(len(self.bid_prices))])/sum(self.bid_volumes)
        self.ask_price_vwap = sum([self.ask_prices[i]*self.ask_volumes[i] for i in range(len(self.ask_prices))]) / sum(self.ask_volumes)

        self.vamp = (self.bid_price_vwap + self.ask_price_vwap) / 2 # volume adjusted mid price
        self.bid_ask_spread = self.ask_prices[0] - self.bid_prices[0]

    def to_series(self):
        cols=['TIME', 'BID', 'BID_VOL', 'ASK', 'ASK_VOL', 'MID', 'IMB', 'WMID']
        data = [self.timestamp, self.best_bid, self.bid_volumes[0], self.best_ask, self.ask_volumes[0], self.mid_price, self.imbalance, self.weighted_mid_price]
        return pd.Series(data=data, index=cols)
    
    def to_df(self):
        cols=['TIME', 'BID', 'BID_VOL', 'ASK', 'ASK_VOL', 'MID', 'IMB', 'WMID']
        data = [[self.timestamp, self.best_bid, self.bid_volumes[0], self.best_ask, self.ask_volumes[0], self.mid_price, self.imbalance, self.weighted_mid_price]]
        return pd.DataFrame(data=data, columns=cols)



class Buffer:
    """
    Buffer used for calculating indicators for streaming data 
    """
    # based on hummingbot buffer implementation. 

    def __init__(self, length: int, dtype=np.int64) -> None:
        self.dtype = dtype
        self.buffer= np.zeros(length, dtype=self.dtype)
        self.length = length
        self.idx = 0
        self.full = False

    def add_sample(self, val):
        self.buffer[self.idx] = val
        self.idx = (self.idx + 1) % self.length
        if not self.full and self.idx == 0:
            self.full = True
    
    def is_empty(self):
        return (not self.full) and (0==self.idx)

    def set_length(self, value):
        """update the length of the Buffer Object"""
        data = self.get_np_array() 

        self.length = value
        self.buffer = np.zeros(value, dtype=self.dtype)
        self.idx = 0
        self.full = False
        for v in data[-value:]:
            self.add_sample(v)
    
    def get_last_value(self):
        if self.is_empty():
            return np.nan
        return self.buffer[self.idx-1]


    def get_np_array(self) -> np.ndarray:
        if not self.full:
            indexes = np.arange(0, stop=self.idx, dtype=np.int16)
        else:
            indexes = np.arange(self.idx, stop=self.idx + self.length, dtype=np.int16) % self.length
        return np.asarray(self.buffer)[indexes]
    
class Indicator:

    def __init__(self, sampling_length: int = 50, processing_length: int = 15, dtype = np.float64):
        self.sampling_buffer = Buffer(sampling_length, dtype=dtype)
        self.processing_buffer = Buffer(processing_length, dtype=dtype)
        self.dtype = dtype
        self.samples_length = 0

    def add_sample(self, value):

        self.sampling_buffer.add_sample(value)
        indicator_val = self.calculate_indicator()
        self.processing_buffer.add_sample(indicator_val)

    def calculate_indicator(self):
        # abstract method
        raise NotImplementedError
    
    def processing_calculation(self):
        """
        returns the average of the processing buffer
        This is what we use for the "indicator value"
        """
        return np.mean(self.processing_buffer.get_np_array())
    

class VolatilityIndicator(Indicator):
    """ Used for calculating sigma2 """
    def __init__(self, sampling_length: int = 20, processing_length: int = 5, dtype=np.float64):
        super().__init__(sampling_length, processing_length, dtype)

    def calculate_indicator(self):
        # calculates sigma2
        np_sampling_buffer = self.sampling_buffer.get_np_array()
        return np.sqrt(np.sum(np.square(np.diff(np_sampling_buffer))) / np_sampling_buffer.size)    
    
    def processing_calculation(self):
        return self.processing_buffer.get_last_value()



class KappaIndicator:
    """Trading intensity
    
    need to address the poisson intensity lambda with which a limit order will be executed
    as a functiuon of its distance (delta) from the mid-price
    
    Want to estimate the intensity of being "hit" by market orders when you have a limit order 
    at distance deltaP from the mid-price. 
    
    For buy order:
    - for every timestamp, record the midprice Pm_t
    - check the timestamp when the price reaches Pm_t - deltaP (for buy order) 
    
    Once this is done for every pair of deltaP and t, you have a corresponding deltaT, giving us a lambda estimate for each price level.

    DeltaP levels: TICK_SIZE_IN_CENTS = 100
    -500, -400, -300, -200, -100, +100, +200, +300, +400, +500
    
    # on update:
    - add midprice for timestamp
    - see if any of the "target midprices" are hit, if so, record deltaT for that price
    """

    class PriceLevel:
        def __init__(self, timestamp, price: int, levels: list, base_indicator) -> None:
            self.base_indicator = base_indicator # the kappa indicator object
            self.timestamp = timestamp
            self.price = round_price(price)
            self.levels = levels # relative to midpoint price

            self.lambdas_left = set(self.levels)

        def record_hit(self, timestamp, level):
            """the midprice hit a given level"""
            level = round_price(level) 
            deltaP = abs(level-self.price) # price level hit

            if deltaP > max(self.levels):
                # print('deltaP too large:', deltaP)
                return 
            
            idx = self.levels.index(deltaP)
            deltaT = timestamp - self.timestamp # time elapsed since hit 
            for i in range(0, idx):
                dP = self.levels[i]
                if dP in self.lambdas_left:
                    self.base_indicator.lambdas[deltaP].append((self.timestamp, deltaT)) # start timestam
                    self.lambdas_left.discard(dP)

            
    def __init__(self, tick_range:int, sampling_length: int = 20) -> None:
        self.levels = list(range(0, (tick_range + 1)*TICK_SIZE_IN_CENTS, TICK_SIZE_IN_CENTS))
        self.sampling_length = sampling_length # time for sampling

        self.lambdas = {level:collections.deque() for level in self.levels} # # tuple: (lambda start timestamp, lambda_estimate)

        self.absolute_price_mapping = collections.defaultdict(collections.deque) # maps absolute price to list of pricelevel objects that are within range 
        self.lambda_average = {level:0 for level in self.levels} # mapping level to average lambda estimates (used if NA samples in timeframe) 
        self.kappa = 0
        self.alpha = 0


    def add_sample(self, book: OrderBook):
        # update indicator with orderbook
        t = book.timestamp
        rounded_mid = round_price(book.mid_price) 
        new_level = self.PriceLevel(timestamp=t, price=rounded_mid, levels=self.levels, base_indicator=self) # create new for the sample
        for dp in self.levels:
            if dp != 0:
                relative_price_opposite = rounded_mid - dp
                while self.absolute_price_mapping[relative_price_opposite] and self.absolute_price_mapping[relative_price_opposite][0].timestamp < t - self.sampling_length: 
                    self.absolute_price_mapping[relative_price_opposite].popleft() # remove expired pricelevel objects
                for pl in self.absolute_price_mapping[relative_price_opposite]:
                    pl.record_hit(t, rounded_mid)
                self.absolute_price_mapping[relative_price_opposite].append(new_level) # add pricelevel reference to the mapper

            relative_price = rounded_mid + dp
            while self.absolute_price_mapping[relative_price] and self.absolute_price_mapping[relative_price][0].timestamp < t - self.sampling_length: 
                self.absolute_price_mapping[relative_price].popleft() # remove expired pricelevel objects
            for pl in self.absolute_price_mapping[relative_price]:
                pl.record_hit(t, rounded_mid)
            self.absolute_price_mapping[relative_price].append(new_level) # add pricelevel reference to the mapper

        self.update_kappa(time=t)

    def update_kappa(self, time):
        """update kappa value based on self.lambdas"""
        min_lambda = 100000
        for level in reversed(self.levels):
            while self.lambdas[level] and self.lambdas[level][0][0] < time - self.sampling_length: 
                self.lambdas[level].popleft()
            if self.lambdas[level]:
                lambda_sum = np.sum([i[1] for i in self.lambdas[level]]) # sum lambda estimates for the given level 
                self.lambda_average[level] = lambda_sum / len(self.lambdas[level]) 
                self.lambda_average[level] = min(self.lambda_average[level], min_lambda)
                min_lambda = self.lambda_average[level]
        lamdas_list = [l for l in self.lambda_average.values()]
        lamdas_adj = [10**-10 if x==0 else x for x in lamdas_list]
        try: 
            plot = False
            # lambdas = 1/lambdas_adj
            lambdas = lamdas_adj[::-1] 
            # lambda = arrivals per time unit
            (new_kappa, new_A), _ = scipy.optimize.curve_fit(lambda dP, k, A: A*np.exp(-k*dP/TICK_SIZE_IN_CENTS),
                                            self.levels,  # independent variable
                                            lambdas,   # dependent data
                                            p0=(self.kappa, self.alpha),
                                            method='dogbox',
                                            bounds=([0,0], [np.inf, np.inf]))
            self.kappa = new_kappa 
            self.alpha = new_A 
            # plot the curve fit, with dots for the lambdas and lines for the curve fit
            if time > 34200 + 300 and plot:
                x_values = np.array(self.levels)
                # reverse x_values so that the curve fit is in the right direction
                x_values = x_values[::-1]
                y_values = np.array(lamdas_adj)
                plt.scatter(x_values, y_values, color='blue', label='lambdas')
                # x_fit = np.linspace(min(x_values), max(x_values), 1000)
                y_fit = []
                for i in range(len(x_values)):
                    y_fit.append(self.alpha*np.exp(-self.kappa*x_values[i]/TICK_SIZE_IN_CENTS))


                plt.plot(x_values, y_fit, color='red', label='curve fit')
                plt.xlabel('price level')
                plt.ylabel('lambda')
                plt.title('Curve fit of lambda vs price level')
                plt.legend()
                plt.show()

        except (RuntimeError, ValueError) as e:
            print('error updating kappa')
            pass


class Market:

    def __init__(self) -> None:
        self.msft_df = pd.read_csv('./data/dataframes/MSFT_df_resampled.csv')
        self.aapl_df = pd.read_csv('./data/dataframes/AAPL_df_resampled.csv')
        self.amzn_df = pd.read_csv('./data/dataframes/AMZN_df_resampled.csv')
        self.goog_df = pd.read_csv('./data/dataframes/GOOG_df_resampled.csv')
        self.intc_df = pd.read_csv('./data/dataframes/INTC_df_resampled.csv')
        self.start_time = 34200000 # 9:30am

        self.end_time = 57600000 # 4:00pm
        
        self.market_depth = 10 # number of levels of market depth to consider

        # parameters for AS
        self.ask_price = float('inf') 
        self.bid_price = 0 
        self.ask_vol = 0  
        self.bid_vol = 0  
        self.ask_time = 0 
        self.bid_time = 0 

        self.order_time = float('inf') # time of last order
        self.bid_time = float('inf') # time of last bid
        self.ask_time = float('inf') # time of last ask

        self.account_balance = 0


        self.position = 0
        self.pnl = 0
        self.target_inventory = 100
        self.state = 'Q'

        self.gamma = GAMMA # risk aversion parameter

        self.order_id = itertools.count(1)

    def save_dfs(self):
        self.msft_df.to_csv('./data/dataframes/MSFT_df_resampled.csv', index=False)
        self.aapl_df.to_csv('./data/dataframes/AAPL_df_resampled.csv', index=False)
        self.amzn_df.to_csv('./data/dataframes/AMZN_df_resampled.csv', index=False)
        self.goog_df.to_csv('./data/dataframes/GOOG_df_resampled.csv', index=False)
        self.intc_df.to_csv('./data/dataframes/INTC_df_resampled.csv', index=False)

    def calulate_intensity(self):

        save = True
        dfs = [self.msft_df, self.aapl_df, self.amzn_df, self.goog_df, self.intc_df]
        # dfs= [self.aapl_df]
        for df in dfs:
            ki = KappaIndicator(tick_range=5, sampling_length=200)
            kappas = []

            for i in range(len(df)):
                row = df.iloc[i]
                time = row['Time']
                bid_prices = [row[f'Bid_Price_{i}'] for i in range(1, self.market_depth+1)]
                bid_sizes = [row[f'Bid_Size_{i}'] for i in range(1, self.market_depth+1)]
                ask_prices = [row[f'Ask_Price_{i}'] for i in range(1, self.market_depth+1)]
                ask_sizes = [row[f'Ask_Size_{i}'] for i in range(1, self.market_depth+1)]

                book = OrderBook(bid_prices=bid_prices, bid_volumes=bid_sizes, ask_prices=ask_prices, ask_volumes=ask_sizes, timestamp=time) 
                ki.add_sample(book=book)
                kappas.append(ki.kappa)
                progress = round(i/len(df), 2)
                print(f'kappa: {ki.kappa}, price: {book.mid_price}, time: {row["datetime"]}, progress: {round(i/len(df), 2)}', end='\r')
            
            df['kappa'] = kappas
            if len(kappas) != len(df):
                save = False

        if save:
            print('saving dataframes...')
            self.save_dfs()
        else: 
            print('not saving dataframes')


    def calculate_sigma2(self, window_length=20*4):
        """add sigma2 value to the dataframes
        default window length is 1 minute 
        """
        dfs = [self.msft_df, self.aapl_df, self.amzn_df, self.goog_df, self.intc_df]
        for df in dfs:
            df['sigma2'] = df['Price'].rolling(window=window_length).std()
        
        self.save_dfs()


    def simulate_session(self, df: pd.DataFrame):
        """ Simulate trading session
        
        df: order book dataframe

        return df containing all trade information
        """


        def format_num(num: float):
            # format number for printing, round to 2 decimal places, and add commas for thousands
            if not num: 
                return '---' 
            return '{:,}'.format(round(num, 2))
        print('Simulating session...')

        session_df = df.copy() 
        session_df['reservation_price'] = -1
        session_df['ask_vol'] = 0
        session_df['bid_vol'] = 0 
        session_df['ask_price'] = float('inf') 
        session_df['bid_price'] = 0 
        session_df['state'] = 'Q'
        session_df['position'] = 0
        session_df['pnl'] = 0
        session_df['bid_time'] = 0
        session_df['ask_time'] = 0
        session_df['order_time'] = 0 

        num_iters = len(session_df) 

        t_0 = session_df['Time'].iloc[0] # start time
        t_f = session_df['Time'].iloc[-1] # end time

        # t_c: time until cancellation of order
        t_c = ORDER_DELAY# 10 seconds 

        # iterate through each row in dataframe
        # for i in range(80, (session_df)):
        self.order_time = session_df['Time'].iloc[0]
        self.bid_time = session_df['Time'].iloc[0]
        self.ask_time = session_df['Time'].iloc[0]
        num_ask_fills = 0
        num_bid_fills = 0
        num_ask_cancelled = 0
        num_bid_cancelled = 0
        num_bid_quotes = 0
        num_ask_quotes = 0

        for i in range(num_iters): 
            # start after 20 seconds
            row = session_df.iloc[i]
            t_i = row['Time']
            bid_prices = [row[f'Bid_Price_{i}'] for i in range(1, self.market_depth+1)]
            bid_sizes = [row[f'Bid_Size_{i}'] for i in range(1, self.market_depth+1)]
            ask_prices = [row[f'Ask_Price_{i}'] for i in range(1, self.market_depth+1)]
            ask_sizes = [row[f'Ask_Size_{i}'] for i in range(1, self.market_depth+1)]
            book = OrderBook(bid_prices=bid_prices, bid_volumes=bid_sizes, ask_prices=ask_prices, ask_volumes=ask_sizes, timestamp=t_i) 
            price = book.vamp # volume adjusted mid price
            sigma2 = row['sigma2']
            sigma2 = sigma2/10000 # convert to cents
            q = self.position / self.target_inventory
            # time_const = ((t_f - t_i) / (t_f - t_0)) 
            time_const = 1
            r = price - q * self.gamma * sigma2 * time_const # reservation price
            spread = self.gamma * sigma2 * time_const + (2/self.gamma) * math.log(1 + (self.gamma/row['kappa']))
            spread = spread*1000 # convert to cents
            new_bid = r - spread / 2
            new_ask = r + spread / 2
            new_bid = round_price(new_bid)
            new_ask = round_price(new_ask)

            new_bid = max(new_bid, book.bid_prices[0])
            new_ask = min(new_ask, book.ask_prices[0])

            #################### Position Sizing Logic ##################################3
            phi_max = PHI_MAX 
            phi_ask = PHI_ASK 
            phi_bid = PHI_BID 
            eta = 0.3
            if self.position < 0: # we want to buy more
                phi_ask = phi_max * math.exp(- eta * abs(self.position))
                phi_bid = abs(self.position) + phi_ask 
            elif self.position > 0: 
                phi_bid = phi_max * math.exp(- eta * self.position)
                phi_ask = abs(self.position) + phi_bid

            bid_vol = int(phi_bid)
            ask_vol = int(phi_ask)

            for vol, price in zip(book.bid_volumes, book.bid_prices):
                # bid prices = prices that we can sell at
                if vol > 0 and price >= self.ask_price:
                    # fill based on volume available
                    amt_filled = min(vol, self.ask_vol)
                    self.ask_vol -= amt_filled
                    self.account_balance += book.best_bid * amt_filled 
                    self.position -= amt_filled
                    if amt_filled > 0: 
                        num_ask_fills += 1
            for vol, price in zip(book.ask_volumes, book.ask_prices):
                if vol > 0 and price <= self.bid_price:
                    amt_filled = min(vol, self.bid_vol)
                    self.bid_vol -= amt_filled
                    self.account_balance -= book.best_ask * amt_filled 
                    self.position += amt_filled
                    if amt_filled > 0:
                        num_bid_fills += 1

            if self.ask_vol > 0 and t_i - self.ask_time > t_c: # ask order timed out
                num_ask_cancelled += 1
                self.ask_vol = 0
                self.ask_price = None
            if self.bid_vol > 0 and t_i - self.bid_time > t_c: # bid order timed out
                num_bid_cancelled += 1
                self.bid_vol = 0
                self.bid_price = None

            if self.bid_vol == 0:
                self.bid_vol = bid_vol
                self.bid_price = new_bid
                self.bid_time = t_i
                num_bid_quotes += 1

            if self.ask_vol == 0:
                self.ask_vol = ask_vol
                self.ask_price = new_ask
                self.ask_time = t_i
                num_ask_quotes += 1



            self.pnl = self.account_balance + self.position * price
 

            # update row in dataframe
            row['state'] = self.state
            row['ask_vol'] = self.ask_vol
            row['bid_vol'] = self.bid_vol
            row['ask_price'] = self.ask_price
            row['bid_price'] = self.bid_price
            row['order_time'] = self.order_time
            row['position'] = self.position
            row['reservation_price'] = r
            row['pnl'] = self.pnl

            # print(f'Position: {self.position}, PnL: {format_num(self.pnl)}, Ask Price: {format_num(self.ask_price)}, Bid Price: {format_num(self.bid_price)}, Ask Vol: {self.ask_vol}, Bid Vol: {self.bid_vol}, Time: {round(t_i, 2)}, Best Bid: {format_num(book.best_bid)}, Best Ask: {format_num(book.best_ask)}, Progress: {round(100 * (i / len(df)), 2)}%, q: {round(q, 2)}, r: {round(r, 2)}, spread: {round(spread, 2)}, bid-ask spread: {round(book.bid_ask_spread, 2)}, midprice: {round(book.mid_price, 2)}, vamp: {round(book.vamp, 2)}, sigma2: {round(sigma2, 2)}')
            session_df.iloc[i] = row
        
        print('Final PnL: ', self.pnl)
        print(f'Number of bid fills: {num_bid_fills}, Number of ask fills: {num_ask_fills}, Number of bid quotes: {num_bid_quotes}, Number of ask quotes: {num_ask_quotes}, Number of bid cancels: {num_bid_cancelled}, Number of ask cancels: {num_ask_cancelled}')
        return session_df
                

mkt = Market()
# mkt.calulate_intensity()
# mkt.calculate_sigma2()
end_time = 1000 
df = mkt.simulate_session(mkt.aapl_df[300:end_time])
# mkt = Market()
# df_amzn = mkt.simulate_session(mkt.amzn_df[300:end_time])
# mkt = Market()
# df_msft = mkt.simulate_session(mkt.msft_df[300:end_time])
# mkt = Market()
# df_goog = mkt.simulate_session(mkt.goog_df[300:end_time])
# mkt = Market()
# df_intc = mkt.simulate_session(mkt.intc_df[300:end_time])

# df.to_csv('test.csv')
# df_amzn.to_csv('test_amzn.csv')
# df_msft.to_csv('test_msft.csv')
# df_goog.to_csv('test_goog.csv')
# df_intc.to_csv('test_intc.csv')

def plot_results(df):

    # create a grid of 5 plots, arranged vertically
    fig, axs = plt.subplots(5, 1, figsize=(10, 25))

    # plot PnL
    axs[0].plot(df['Time'], df['pnl'])
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('PnL')
    axs[0].set_title('PnL vs Time')

    # plot position
    axs[1].plot(df['Time'], df['position'])
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Position')
    axs[1].set_title('Position vs Time')

    # plot price
    # add bid and ask prices
    axs[2].plot(df['Time'], df['ask_price'], color='red')
    axs[2].plot(df['Time'], df['bid_price'], color='green')
    # plot actual bid and ask prices
    axs[2].plot(df['Time'], df['Ask_Price_1'], color='blue', linestyle='dashed')
    axs[2].plot(df['Time'], df['Bid_Price_1'], color='black', linestyle='dashed')

    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Price')
    axs[2].set_title('Price vs Time')


    # plot sigma2
    axs[3].plot(df['Time'], df['sigma2'])
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('sigma2')
    axs[3].set_title('sigma2 vs Time')

    # plot intensity
    axs[4].plot(df['Time'], df['kappa'])
    axs[4].set_xlabel('Time')
    axs[4].set_ylabel('Intensity')
    axs[4].set_title('Intensity vs Time')


    # Display the figure with all the subplots
    plt.tight_layout()
    plt.show()

    # # plot bid and ask prices on single large plot
    # fig, ax = plt.subplots(figsize=(30, 10))
    # ax.plot(df['Time'], df['ask_price'], color='red')
    # ax.plot(df['Time'], df['bid_price'], color='green')
    # ax.plot(df['Time'], df['Ask_Price_1'], color='blue', linestyle='dashed')
    # ax.plot(df['Time'], df['Bid_Price_1'], color='black', linestyle='dashed')
    # ax.plot(df['Time'], df['Price'], color='orange') 
    # ax.legend(['Ask Price', 'Bid Price', 'Actual Ask Price', 'Actual Bid Price', 'Mid Price'])
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Price')
    # ax.set_title('Price vs Time')
    # plt.show()


plot_results(df)
# plot_results(df_amzn)
# plot_results(df_msft)
# plot_results(df_goog)
# plot_results(df_intc)







# %%
