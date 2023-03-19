# Skript to compare some models to find the best suitable one (benchmark)

# Import the external packages
# Operating system functionalities
import sys
import os
from pathlib import Path
# For ExponentialSmoothing Forecasting
from statsmodels.tsa.api import ExponentialSmoothing
# For auto_arima model
import pmdarima as pm
# For LinearRegression
from sklearn.linear_model import LinearRegression
# Interface for time series learning tasks
from sktime.forecasting.all import ForecastingHorizon, EnsembleForecaster
from sktime.forecasting.compose import make_reduction
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

def figure_time_series_data_as_layers(figure_title, y_label, x_data, y_layers, y_datas):
    """
    Function to create a figure for time series data as multiple layers
    ----------
    Parameters:
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
        The figure
    """

    try:
        own_logger.info("Figure for times series data as multiple layers with title %s", figure_title)
        figure = PlotMultipleLayers(figure_title, "date", y_label, x_axis_type='datetime')
        for (index, layer) in enumerate(y_layers):
            own_logger.info("Add CircleLayer for %s", layer)
            figure.addCircleLayer(layer, x_data, y_datas[y_datas.columns[index]])
        return figure.getFigure()
    except TypeError as error:
        own_logger.error("########## Error when trying to create figure ##########", exc_info=error)
        sys.exit('A parameter does not match the given type')

#########################################################

class Model():
    """
    A parent class for a model
    ----------
    Attributes:
        no attributes
    ----------
    Methods
        evaluate: Method for model evaluation
    """
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

class AutoArimaModel(Model):
    """
    A class for a auto_arima model (multivariate)
    ----------
    Attributes:
        no attributes
    ----------
    Methods
        train: Method to train the model
        predict: Method for time series prediction
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
        self.__model = pm.auto_arima(y_train.iloc[:,1:], X_train.iloc[:,1:], seasonal=False, stationary=False, test='kpss', stepwise=True, trace=True)
        # Trial: Only Univariate
        #self.__model = pm.auto_arima(y_train.iloc[:,1:], seasonal=True, m=365, stationary=False, test='kpss', stepwise=True, trace=True)
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

#########################################################

class ExponentialSmoothingModel(Model):
    """
    A class for a ExponentialSmoothing (univariate) model
    ----------
    Attributes:
        no attributes
    ----------
    Methods
        train: Method to train the model
        predict: Method for time series prediction
    """

    # Constructor Method
    def __init__(self):
        own_logger.info("Initialize a ExponentialSmoothing model")
        # TODO: Intializations?

    def train(self, y_train, consider_trend=True):
        """
        Train the model
        ----------
        Parameters:
            y_train : Series
                The y data
            consider_trend : Boolean
                Should a trend be considered?
        ----------
        Returns:
            no returns
        """
        # Train the model
        own_logger.info("Train the model")
        # ExponentialSmoothing-Model with additive decomposition with daily data with a yearly cycle (seasonal_periods=365)
        # TODO: Seasonal period should be detected automatically?
        #exp = ExponentialSmoothing(y_train.iloc[:,1:], trend='additive', seasonal='additive')
        if consider_trend:
            # With a trend
            exp = ExponentialSmoothing(y_train.iloc[:,1:], trend='additive', seasonal='additive', seasonal_periods=365)
        else:
            # Without a trend
            exp = ExponentialSmoothing(y_train.iloc[:,1:], seasonal='additive', seasonal_periods=365)
        self.__model = exp.fit(use_brute=True, optimized=True)

    def predict(self, len):
        """
        Time Series Forecast
        ----------
        Parameters:
            len : int
                The length of the series to predict
        ----------
        Returns:
            The predictions as Series
        """
        # Prediction
        own_logger.info("Prediction of %d values", len)
        y_hat = self.__model.forecast(len)
        return y_hat
    
#########################################################

class LinearRegressionModel(Model):
    """
    A class for a LinearRegression model
    ----------
    Attributes:
        no attributes
    ----------
    Methods
        train: Method to train the model
        predict: Method for time series prediction
    """

    # Constructor Method
    def __init__(self, y_test):
        own_logger.info("Initialize a LinearRegression model")
        # TODO: Intializations?
        self.__fh = ForecastingHorizon(y_test.index, is_relative=False)

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
        # LinearRegression-Model
        regressors = [
            ("LinearRegression", make_reduction(LinearRegression()))]
        self.__model = EnsembleForecaster(regressors)
        self.__model.fit(y=y_train.iloc[:,1:], X=X_train.iloc[:,1:])

    def predict(self, X_test):
        """
        Time Series Forecast
        ----------
        Parameters:
            X_test : DataFrame
                The x data
        ----------
        Returns:
            The predictions as Series
        """
        # Prediction
        own_logger.info("Prediction")
        y_hat = self.__model.predict(X=X_test, fh=self.__fh)
        return y_hat

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

    #Set datetime as index
    y_train.set_index(y_train.date, inplace=True)
    X_train.set_index(X_train.date, inplace=True)
    y_test.set_index(y_test.date, inplace=True)
    X_test.set_index(X_test.date, inplace=True)
    # Set data frequency to day
    y_train.index.freq='D'
    X_train.index.freq='D'
    y_test.index.freq='D'
    X_test.index.freq='D'

    # Model 1: ExponentialSmoothing
    own_logger.info("########## Model 1: Classic Statistical ExponentialSmoothing (univariate, only consider the target variable) ##########")
    # Train the model
    own_logger.info("########## Train the Model 1 ##########")
    # Model with the consideration of a trend
    __model = ExponentialSmoothingModel()
    # replace zero values?
    #y_train_non_zeros = y_train.replace(to_replace=0, method='ffill')
    #__model.train(y_train_non_zeros, False)
    __model.train(y_train)
    # Model without the consideration of a trend
    __model_no_trend = ExponentialSmoothingModel()
    __model_no_trend.train(y_train, False)
    # Forecast
    own_logger.info("########## Forecasting Model 1 ##########")
    y_hat = __model.predict(len(y_test))
    y_hat_no_trend = __model_no_trend.predict(len(y_test))

    # Visualize the forecast data
    # With trend
    # Merge the y_test with the y_hat data
    # replace zero values?
    #df_y = y_test.replace(to_replace=0, method='ffill').copy()
    df_y = y_test.copy()
    df_y[y_test.sby_need.name + "_pred"] = y_hat.values
    own_logger.info("########## Visualize the forcast data Model 1: With trend ##########")
    # Create dict to define which data should be visualized as layers
    dict_figures = {
        "label": df_y.columns.values[1:],   # Skip the first column which includes the date (represents the x-axis)
    }
    figure_1_a = figure_time_series_data_as_layers("ExponentialSmoothing with Trend", "sby_need", df_y.date, df_y[dict_figures.get('label')].columns.values, df_y[dict_figures.get('label')])
    # Without trend
    # Merge the y_test with the y_hat_no_trend data
    df_y = y_test.copy()
    df_y[y_test.sby_need.name + "_pred"] = y_hat_no_trend.values
    own_logger.info("########## Visualize the forcast data Model 1: Without trend ##########")
    # Create dict to define which data should be visualized as layers
    dict_figures = {
        "label": df_y.columns.values[1:],   # Skip the first column which includes the date (represents the x-axis)
    }
    figure_1_b = figure_time_series_data_as_layers("ExponentialSmoothing without Trend", "sby_need", df_y.date, df_y[dict_figures.get('label')].columns.values, df_y[dict_figures.get('label')])

    # Metrics for Evaluation: Model without Trend
    MSE, MAE, RMSE, MAPE, R2 = __model_no_trend.evaluate(y_test.iloc[:,1:], y_hat_no_trend)
    own_logger.info("########## Evaluation Metrics of the Model 1 without Trend ##########")
    own_logger.info("MSE = %f", MSE)
    own_logger.info("MAE = %f", MAE)
    own_logger.info("RMSE = %f", RMSE)
    own_logger.info("MAPE = %f", MAPE)
    own_logger.info("R2 = %f", R2)

    # Model 2: auto_arima
    own_logger.info("########## Model 2: Classic Statistical auto_arima (multivariate) ##########")
    # Train the model
    own_logger.info("########## Train the Model 1 ##########")
    __model = AutoArimaModel()
    # Second approach: Only use the target value and the high correlated variable "calls"
    __model.train(X_train.loc[:,[X_train.date.name, X_train.calls.name]], y_train)

    # Merge the test data for prediction
    # Second approach: Only use the target value and the high correlated variable "calls"
    df_test = X_test.loc[:,[X_test.date.name, X_test.calls.name]]
    df_test[y_test.sby_need.name] = y_test.loc[:,y_test.sby_need.name]
    # Forecast
    own_logger.info("########## Forecasting model 2 ##########")
    #y_hat, conf_interval = __model.predict(len(df_test), X_test.loc[:,~X_test.columns.isin(["date"])])
    # Second approach: Only use the target value and the high correlated variable "calls" (exegenous must be a 2D array)
    y_hat, conf_interval = __model.predict(len(df_test), X_test.loc[:,X_test.calls.name].values.reshape((len(X_test.loc[:,X_test.calls.name]),1)))

    # Visualize the forecast data
    # Merge the y_test with the y_hat data
    df_y = y_test.copy()
    df_y[y_test.sby_need.name + "_pred"] = y_hat
    own_logger.info("########## Visualize the forcast data Model 2 ##########")
    # Create dict to define which data should be visualized as layers
    dict_figures = {
        "label": df_y.columns.values[1:],   # Skip the first column which includes the date (represents the x-axis)
    }
    figure_2 = figure_time_series_data_as_layers("AutoArimaModel", "sby_need", df_y.date, df_y[dict_figures.get('label')].columns.values, df_y[dict_figures.get('label')])

    # Metrics for Evaluation
    MSE, MAE, RMSE, MAPE, R2 = __model.evaluate(y_test.iloc[:,1:], y_hat)
    own_logger.info("########## Evaluation Metrics of the Model 2: ##########")
    own_logger.info("MSE = %f", MSE)
    own_logger.info("MAE = %f", MAE)
    own_logger.info("RMSE = %f", RMSE)
    own_logger.info("MAPE = %f", MAPE)
    own_logger.info("R2 = %f", R2)

    # Model 3: LinearRegression (Supervised ML Model)
    own_logger.info("########## Model 3: Supervised ML model: LinearRegression (multivariate) ##########")
    # Train the model
    own_logger.info("########## Train the Model 3 ##########")
    __model = LinearRegressionModel(y_test)
    # Second approach: Only use the target value and the high correlated variable "calls"
    __model.train(X_train.loc[:,[X_train.date.name, X_train.calls.name]], y_train)

    # Forecast
    own_logger.info("########## Forecasting model 3 ##########")
    y_hat = __model.predict(X_test.loc[:,X_train.calls.name])

    # Visualize the forecast data
    # Merge the y_test with the y_hat data
    df_y = y_test.copy()
    df_y[y_test.sby_need.name + "_pred"] = y_hat
    own_logger.info("########## Visualize the forcast data Model 3 ##########")
    # Create dict to define which data should be visualized as layers
    dict_figures = {
        "label": df_y.columns.values[1:],   # Skip the first column which includes the date (represents the x-axis)
    }
    figure_3 = figure_time_series_data_as_layers("LinearRegression", "sby_need", df_y.date, df_y[dict_figures.get('label')].columns.values, df_y[dict_figures.get('label')])

    # Metrics for Evaluation
    MSE, MAE, RMSE, MAPE, R2 = __model.evaluate(y_test.iloc[:,1:], y_hat)
    own_logger.info("########## Evaluation Metrics of the Model 3: ##########")
    own_logger.info("MSE = %f", MSE)
    own_logger.info("MAE = %f", MAE)
    own_logger.info("RMSE = %f", RMSE)
    own_logger.info("MAPE = %f", MAPE)
    own_logger.info("R2 = %f", R2)

    # Create the plot with the created figures
    file_name = "model_benchmark.html"
    file_title = "Model Benchmark"
    own_logger.info("Plot times series data with title %s as multiple figures to file %s", file_title, file_name)
    plot = PlotMultipleFigures(os.path.join("output",file_name), file_title)
    plot.appendFigure(figure_1_a)
    plot.appendFigure(figure_1_b)
    plot.appendFigure(figure_2)
    plot.appendFigure(figure_3)
    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')
