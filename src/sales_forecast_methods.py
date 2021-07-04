import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from sklearn.preprocessing import MinMaxScaler
#from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import gru_model


gru_model.seed_all(10)


def forecast_sales(start_date: datetime,
                   end_date: datetime,
                   sales_data: pd.DataFrame):
    """ Mean-based approach (current) for sales forecasting

    :param start_date: init date
    :param end_date: end date
    :param sales_data: previous sales
    :returns estimated sales
    """
    days_to_forecast = (end_date - start_date).days
    daily_sales = sales_data[sales_data['date'] < start_date]['sales']
    #daily_sales.hist()
    return daily_sales.mean() * days_to_forecast


def calculate_new_order(order_date: datetime,
                        lead_time_days: int,
                        days_to_next_order: int,
                        sales_data: pd.DataFrame,
                        current_stock_level: int,
                        stock_in_transit: int):
    """ Method that estimates next order based on the mean (current) approach

    :param order_date: the order date
    :param lead_time_days: days to get the order
    :param days_to_next_order: time in days to next order
    :param sales_data: previous sales data
    :param current_stock_level: the current stock available
    :param stock_in_transit: the stock in transit
    :returns estimated order
    """
    arrival_date = order_date + timedelta(days=lead_time_days)
    next_arrival_date = order_date + timedelta(days=days_to_next_order + lead_time_days)

    forecast_leadtime = forecast_sales(order_date, arrival_date, sales_data)
    #print(forecast_leadtime)
    estimated_stock_at_arrival = current_stock_level - forecast_leadtime + stock_in_transit
    forecast_planning_horizon = forecast_sales(arrival_date, next_arrival_date, sales_data)
    #print(forecast_planning_horizon)

    order = max(0, forecast_planning_horizon - estimated_stock_at_arrival)
    return order


def forecast_sales_seasonal(start_date: datetime,
                            end_date: datetime,
                            sales_data: pd.DataFrame):
    """ Seasonal-based approach for sales forecasting

    :param start_date: init date
    :param end_date: end date
    :param sales_data: previous sales
    :returns estimated sales
    """
    #days_to_forecast = (end_date - start_date).days
    # select mean daily sales but for same season
    sales_data['date'] = pd.to_datetime(sales_data['date'], format='%Y%m%d')
    month = start_date.month
    day = start_date.day + 1
    if start_date.day >= 30:
        month += 1
        day = 0
    daily_sales_season = sales_data.loc[(sales_data['date'].dt.year < start_date.year) &
                                        (sales_data['date'].dt.month >= month) &
                                        (sales_data['date'].dt.month <= end_date.month) &
                                        (sales_data['date'].dt.day >= day) &
                                        (sales_data['date'].dt.day <= end_date.day)]
    daily_sales_season.index = daily_sales_season['date']
    daily_sales_season = daily_sales_season.resample('1Y').sum()
    #daily_sales_season.hist()

    return np.mean(daily_sales_season.values) # * days_to_forecast


def calculate_new_order_seasonal(order_date: datetime,
                                 lead_time_days: int,
                                 days_to_next_order: int,
                                 sales_data: pd.DataFrame,
                                 current_stock_level: int,
                                 stock_in_transit: int):
    """ Method that estimates next order based on the seasonal approach

    :param order_date: the order date
    :param lead_time_days: days to get the order
    :param days_to_next_order: time in days to next order
    :param sales_data: previous sales data
    :param current_stock_level: the current stock available
    :param stock_in_transit: the stock in transit
    :returns estimated order
    """
    arrival_date = order_date + timedelta(days=lead_time_days)
    next_arrival_date = order_date + timedelta(days=days_to_next_order + lead_time_days)

    forecast_leadtime = forecast_sales_seasonal(order_date, arrival_date, sales_data)
    #print(forecast_leadtime)
    estimated_stock_at_arrival = current_stock_level - forecast_leadtime + stock_in_transit
    forecast_planning_horizon = forecast_sales_seasonal(arrival_date, next_arrival_date, sales_data)
    #print(forecast_planning_horizon)

    order = max(0, forecast_planning_horizon - estimated_stock_at_arrival)
    return order


def forecast_sales_recurrent(start_date: datetime,
                             end_date: datetime,
                             sales_data: pd.DataFrame,
                             verbose: int = 1):
    """ Recurrent-based approach for sales forecasting

    :param start_date: init date
    :param end_date: end date
    :param sales_data: previous sales
    :returns estimated sales
    """
    days_to_forecast = (end_date - start_date).days
    input_dim = days_to_forecast
    # prepare data with the time lags
    target = sales_data['sales'].values.reshape(-1, 1)
    scaler = MinMaxScaler().fit(target)
    target = scaler.transform(target)
    # export current scaler to file for further use
    pickle.dump(scaler, open('scaler_' + str(start_date) + '_' + str(days_to_forecast) + '.p', 'wb'))
    df_generated = gru_model.generate_time_lags(pd.DataFrame(target, columns=['sales']), input_dim)

    val_ratio = 0.2

    X = df_generated.drop('sales', axis=1)
    y = df_generated['sales']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, shuffle=False)

    batch_size = min(30, input_dim)

    train_features = torch.Tensor(X_train.values)
    train_targets = torch.Tensor(y_train.values)
    val_features = torch.Tensor(X_val.values)
    val_targets = torch.Tensor(y_val.values)

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)

    # train the model
    input_dim = len(X_train.columns)
    output_dim = 1
    hidden_dim = 128
    layer_dim = 3
    batch_size = batch_size
    dropout = 0.2
    n_epochs = 100
    learning_rate = 1e-3
    weight_decay = 1e-6

    model_params = {'input_dim': input_dim,
                    'hidden_dim' : hidden_dim,
                    'layer_dim' : layer_dim,
                    'output_dim' : output_dim,
                    'dropout_prob' : dropout}

    model = gru_model.GRUModel(**model_params)

    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    opt = gru_model.Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
    opt.train(train_loader,
              val_loader,
              batch_size=batch_size,
              n_epochs=n_epochs,
              n_features=input_dim,
              model_name='gru_' + str(start_date) + '_' + str(days_to_forecast) + '.pt',
              verbose=verbose)
    #opt.plot_losses()

    # make predictions based on previous trend
    d = timedelta(days=days_to_forecast)
    X_test = sales_data[(sales_data['date'] < start_date) &
                        (sales_data['date'] >= start_date-d)]
    if X_test.shape[0] < days_to_forecast:
        year = start_date.year - 1
        X_test = sales_data[(sales_data['date'].dt.year == year)]
        month = (start_date-d).month
        day = (start_date-d).day
        if (start_date-d).day >= 30:
            month += 1
            day = 0
        X_test = X_test[(X_test['date'].dt.month >= month) &
                        (X_test['date'].dt.month <= start_date.month) &
                        (X_test['date'].dt.day >= day) &
                        (X_test['date'].dt.day <= start_date.day)]
    test = np.array(X_test['sales']).reshape(-1,1)
    test = scaler.transform(test)

    preds = []
    # prediction step using recurrence
    x_rnn = torch.Tensor(test).squeeze()
    y_hat = x_rnn[-1]
    preds_tmp = []
    for n in range(days_to_forecast):
        if n>0:
            x_rnn = torch.cat((x_rnn[1:], torch.tensor([y_hat]).float()), 0)
        x_test = x_rnn.view([1, -1, input_dim]).to(gru_model.device)
        opt.model.eval()
        yhat = opt.model(x_test)
        preds_tmp.append(yhat.cpu().detach().numpy()[0][0])
    preds = np.concatenate((preds, preds_tmp))
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    #plt.figure()
    #plt.hist(preds)

    return np.sum(preds) # * days_to_forecast


def calculate_new_order_recurrent(order_date: datetime,
                                  lead_time_days: int,
                                  days_to_next_order: int,
                                  sales_data: pd.DataFrame,
                                  current_stock_level: int,
                                  stock_in_transit: int):
    """ Method that estimates next order based on the recurrent approach

    :param order_date: the order date
    :param lead_time_days: days to get the order
    :param days_to_next_order: time in days to next order
    :param sales_data: previous sales data
    :param current_stock_level: the current stock available
    :param stock_in_transit: the stock in transit
    :returns estimated order
    """
    arrival_date = order_date + timedelta(days=lead_time_days)
    next_arrival_date = order_date + timedelta(days=days_to_next_order + lead_time_days)

    forecast_leadtime = forecast_sales_recurrent(order_date, arrival_date, sales_data, verbose=0)
    #print(forecast_leadtime)
    estimated_stock_at_arrival = current_stock_level - forecast_leadtime + stock_in_transit

    forecast_planning_horizon = forecast_sales_recurrent(arrival_date, next_arrival_date, sales_data, verbose=0)
    #print(forecast_planning_horizon)

    order = max(0, forecast_planning_horizon - estimated_stock_at_arrival)
    return order