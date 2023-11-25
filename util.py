import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error

def plot_gene(df, gene):
    yaxis = df.iloc[:,gene]
    xaxix = df.index

    plt.plot(xaxix, yaxis, marker='o')

    plt.xlabel(xaxix.name)
    plt.tick_params(axis='x',rotation=90)
    plt.ylabel('level')
    plt.title(yaxis.name)

    plt.show()
# plot_gene(df, gene = 100)

def plot_all(df):
    xaxix = df.index
    # plt.xlabel(xaxix.name)
    for col in df.columns:
        yaxis = df.loc[:,col]
        # plt.plot(xaxix, yaxis, marker='o')
        plt.plot(xaxix, yaxis)
    plt.show()
# plot_all(df)

def timeseries_generator(data: pd.core.frame.DataFrame, time_steps, b_size):
    X = []
    y = []
    generator_data = TimeseriesGenerator(data.values, data.values, length=time_steps, batch_size=b_size)
    for i in range(len(generator_data)):
        _X, _y = generator_data[i]
        X.extend(_X)
        y.extend(_y)
    X = np.array(X)
    y = np.array(y).flatten()
    return X, y
# X, y = timeseries_generator(df, 5)

def plot_metrics_history(metrics_names, history):
    fig, axes = plt.subplots(nrows=len(metrics_names), ncols=1, figsize=(20, 10))
    _ = fig.suptitle('Metrics')

    for i in range(len(metrics_names)):
        metric = metrics_names[i]
        _ = axes[i].plot(history.history[metric], label=metric)
        _ = axes[i].plot(history.history[f'val_{metric}'], label=f'val_{metric}')
        _ = axes[i].legend()
# plot_metrics_history(model.metrics_names, history)

def plot_gene_pred(df, y_test, y_predicted, gene):
    test_time_point = y_predicted.shape[0]
    yaxis = df.iloc[:,gene]
    xaxis = df.index
    train_point_x = xaxis[:-test_time_point]
    train_point_y = yaxis[:-test_time_point]

    # plt.plot(xaxis, yaxis, marker='o')
    plt.plot(train_point_x, train_point_y, marker='o', color='black', label='train')
    plt.plot(xaxis[-test_time_point:],y_test[:,gene], marker='o', color='green', label='test')
    plt.plot(xaxis[-test_time_point:],y_predicted[:,gene], marker='o', color='red', label='predicted')

    plt.xlabel(xaxis.name)
    plt.tick_params(axis='x',rotation=90)
    plt.ylabel(df.columns.name)
    plt.title(yaxis.name)

    # plt.show()
    plt.legend()

def get_gene_pred_info(y_test, y_predicted, feature):
    print(f'for feature {feature}')
    print(f'test        {y_test[:,feature]}')
    print(f'predictions {y_predicted[:,feature]}')
    rmse = mean_squared_error(y_test[:,feature], y_predicted[:,feature])
    print('Test RMSE: %.3f' % rmse)