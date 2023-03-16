# Skript to compare some models to find the best suitable one (benchmark)

# Import the external packages
# Operating system functionalities
import sys
import os
from pathlib import Path
# For auto_arima model
import pmdarima as pm
# For evaluation metrics
from sklearn import metrics
# For various calculations
import numpy as np

# Import internal packages/ classes
# Import the src-path to sys path that the internal modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
# To handle the Logging for all modules in the same way
from utils.own_logging import OwnLogging
# To plot data with bokeh
from utils.plot_data import PlotMultipleLayers, PlotMultipleFigures
# To handle csv files
from utils.csv_operations import load_data, save_data, convert_date_in_data_frame

#########################################################

# Initialize the logger
own_logger = OwnLogging(Path(__file__).stem).logger

#########################################################

def plot_time_series_data_as_layers(file_name, file_title, figure_title, y_label, x_data, y_layers, y_datas):
    """
    Function to plot time series data as multiple layers
    ----------
    Parameters:
        file_name : str
            The file name (html) in which the figure is shown
        file_title : str
            The output file title
        figure_title : str
            The title of the figure
        y_label : str
            The label of the y axis
        x_data : Series
                The x data to plot
        y_layers : array
            The names of the layers
        y_datas : DataFrame
            The y data to plot
    ----------
    Returns:
        no returns
    """

    try:
        own_logger.info("Plot times series data with title %s as multiple layers to file %s", file_title, file_name)
        figure_plot = PlotMultipleLayers(figure_title, "date", y_label, x_axis_type='datetime', file_name=os.path.join("output",file_name), file_title=file_title)
        for (index, layer) in enumerate(y_layers):
            own_logger.info("Add CircleLayer for %s", layer)
            figure_plot.addCircleLayer(layer, x_data, y_datas[y_datas.columns[index]])
        # Show the plot in responsive layout, but only stretch the width
        figure_plot.showPlotResponsive('stretch_width')
    except TypeError as error:
        own_logger.error("########## Error when trying to plot data ##########", exc_info=error)
        sys.exit('A parameter does not match the given type')

#########################################################

class AutoArimaModel():
    """
    A class for a auto_arima model
    ----------
    Attributes:
        file_name : str
            The file name (html) in which the figure is shown
        file_title : str
            The output file title
    ----------
    Methods
        no methods
    """

    # Constructor Method
    def __init__(self):
        own_logger.info("Initialize a auto arima model")
        # TODO: Intializations?

    def train(self, X_train, y_train):
        """
        Train the model
        ----------
        Parameters:
            X_train : DataFrame
                The x data
            y_train : Series
                The y data
        ----------
        Returns:
            no returns
        """
        # Train the model
        own_logger.info("Train the model")
        # auto_arima with daily data with a yearly cycle (seasonal_periods=365)
        #self.__model = pm.auto_arima(y_train.iloc[:,1:], X_train.iloc[:,1:], seasonal=True, m=365, stationary=False, test='kpss', stepwise=True, trace=True)
        # Big values for seasonal period takes too long, for first tests use simpler approach, so no seasonality: TODO!
        self.__model = pm.auto_arima(y_train.iloc[:,1:], X_train.iloc[:,1:], seasonal=True, m=1, stationary=False, test='kpss', stepwise=True, trace=True)
        own_logger.info("Best model: ARIMA%s%s", self.__model.order, self.__model.seasonal_order)
        # Get summary of finding the best hyperparameters
        #self.__model.summary()

    def predict(self, len, exogenous):
        """
        Time Series Forecast
        ----------
        Parameters:
            len : int
                The length of the series to predict
            exogenous : DataFrame
                Exogenous array for prediction
        ----------
        Returns:
            The predictions as Series and the confidence interval
        """
        # Prediction
        own_logger.info("Prediction of %d values", len)
        y_hat, conf_interval = self.__model.predict(len, exogenous=exogenous, return_conf_int=True)
        return y_hat, conf_interval
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate the time series forecast
        ----------
        Parameters:
            y_true : Series
                The true values
            y_pred : Series
                The predicted values
        ----------
        Returns:
            The evaluation metrics as float:
                MSE, MAE, RMSE, MAPE, R2
        """
        own_logger.info("Calculate various metrics for evaluation")
        # MSE
        MSE = metrics.mean_squared_error(y_true, y_pred)
        own_logger.info("MSE = %f", MSE)
        # MAE
        MAE = metrics.mean_absolute_error(y_true, y_pred)
        own_logger.info("MAE = %f", MAE)
        # RMSE
        RMSE = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
        own_logger.info("RMSE = %f", RMSE)
        # MAPE
        MAPE = metrics.mean_absolute_percentage_error(y_true, y_pred)
        own_logger.info("MAPE = %f", MAPE)
        #R2
        R2 = metrics.r2_score(y_true, y_pred)
        own_logger.info("R2 = %f", R2)

        return MSE, MAE, RMSE, MAPE, R2

#########################################################
#########################################################
#########################################################

# When this script is called directly...
if __name__ == "__main__":
    # ...then calling the functions

    own_logger.info("########## START ##########")

    # Loading the featurized data as pandas DataFrame
    own_logger.info("########## Loading the training data ##########")
    X_train = load_data("modeling", "train_input.csv")
    y_train = load_data("modeling", "train_target.csv")
    X_test = load_data("modeling", "test_input.csv")
    y_test = load_data("modeling", "test_target.csv")

    # Convert the date
    own_logger.info("########## Convert the column date to DateTime ##########")
    convert_date_in_data_frame(X_train)
    convert_date_in_data_frame(y_train)
    convert_date_in_data_frame(X_test)
    convert_date_in_data_frame(y_test)

    #Set datetime as index so that the frequency of the data can be automatically detected?
    # TODO: Does not work maybe due to missing rows (deleted outliers)?
    #y_train.set_index(y_train.date, inplace=True)

    # Train the model
    own_logger.info("########## Train the classic statistical model: auto_arima ##########")
    __model = AutoArimaModel()
    # Exclude constant column
    __model.train(X_train.loc[:,X_train.columns!="n_sby"], y_train)

    # Merge the test data for prediction without columns date and n_sby
    df_test = X_test.loc[:,~X_test.columns.isin(["date", "n_sby"])]
    df_test[y_test.sby_need.name] = y_test.loc[:,y_test.sby_need.name]
    # Forecast
    own_logger.info("########## Forecasting ##########")
    y_hat, conf_interval = __model.predict(len(df_test), X_test.loc[:,~X_test.columns.isin(["date", "n_sby"])])

    # Visualize the forecast data
    # Merge the y_test with the y_hat data
    df_y = y_test.copy()
    df_y[y_test.sby_need.name + "_pred"] = y_hat
    own_logger.info("########## Visualize the forcast data ##########")
    # Create dict to define which data should be visualized as layers
    dict_figures = {
        "label": df_y.columns.values[1:],   # Skip the first column which includes the date (represents the x-axis)
    }
    plot_time_series_data_as_layers("model_benchmark.html", "Model Benchmark", "AutoArimaModel", "sby_need", df_y.date, df_y[dict_figures.get('label')].columns.values, df_y[dict_figures.get('label')])

    # Metrics for Evaluation
    MSE, MAE, RMSE, MAPE, R2 = __model.evaluate(y_test.iloc[:,1:], y_hat)
    own_logger.info("########## Evaluation Metrics of the Model: ##########")
    own_logger.info("MSE = %f", MSE)
    own_logger.info("MAE = %f", MAE)
    own_logger.info("RMSE = %f", RMSE)
    own_logger.info("MAPE = %f", MAPE)
    own_logger.info("R2 = %f", R2)
