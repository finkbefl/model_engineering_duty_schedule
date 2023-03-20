# Skript for training a baseline model (Any model that is worse than this is not worth being considered at all)

# Import the external packages
# Operating system functionalities
import sys
import os
from pathlib import Path
# For Seasonal Naïve Forecasting
from statsforecast.models import SeasonalNaive
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
            own_logger.info("Add Layer for %s", layer)
            figure_plot.addLineCircleLayer(layer, x_data, y_datas[y_datas.columns[index]])
        # Show the plot in responsive layout, but only stretch the width
        figure_plot.showPlotResponsive('stretch_width')
    except TypeError as error:
        own_logger.error("########## Error when trying to plot data ##########", exc_info=error)
        sys.exit('A parameter does not match the given type')

#########################################################

class SeasonalNaiveModel():
    """
    A class for a baseline model: SeasonalNaive (univariate, only consider the target variable)
    ----------
    Attributes:
        no attributes
    ----------
    Methods
        train: Method to train the model
        predict: Method for time series prediction
        evaluate: Method for evaluation
    """

    # Constructor Method
    def __init__(self):
        own_logger.info("Initialize a baseline model: SeasonalNaive")
        # TODO: Intializations?

    def train(self, y_train):
        """
        Train the model
        ----------
        Parameters:
            y_train : Series
                The y data
        ----------
        Returns:
            no returns
        """
        # Train the baseline model
        own_logger.info("Train the baseline model")
        # SeasonalNaive-Model with daily data with a yearly cycle (season_length=365)
        seasonal_naive = SeasonalNaive(season_length=365)
        # Train the model (Note: The train data must be a 1D numpy array!)
        self.__model = seasonal_naive.fit(y=y_train.iloc[:,1:].values.ravel())

    def predict(self, len):
        """
        Time Series Forecast
        ----------
        Parameters:
            len : int
                The length of the series to predict
        ----------
        Returns:
            The predictions as numpy array
        """
        # Prediction
        own_logger.info("Prediction of %d values", len)
        y_hat = self.__model.predict(len)
        # return the point predictions out of the dict
        return y_hat['mean']
    
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

    # Loading the featurized data as pandas DataFrame: Only the with the target variable (univariate)
    own_logger.info("########## Loading the training data ##########")
    #X_train = load_data("modeling", "train_input.csv")
    y_train = load_data("modeling", "train_target.csv")
    #X_test = load_data("modeling", "test_input.csv")
    y_test = load_data("modeling", "test_target.csv")

    # Convert the date
    own_logger.info("########## Convert the column date to DateTime ##########")
    #convert_date_in_data_frame(X_train)
    convert_date_in_data_frame(y_train)
    #convert_date_in_data_frame(X_test)
    convert_date_in_data_frame(y_test)

    # Train the baseline model: SeasonalNaive
    own_logger.info("########## Train the baseline model: SeasonalNaive (univariate, only consider the target variable) ##########")
    __baseline_model = SeasonalNaiveModel()
    __baseline_model.train(y_train)

    # Forecast
    own_logger.info("########## Forecasting ##########")
    y_hat = __baseline_model.predict(len(y_test))

    # Visualize the forecast data
    # Merge the y_test with the y_hat data
    df_y = y_test.copy()
    df_y[y_test.sby_need.name + "_pred"] = y_hat
    own_logger.info("########## Visualize the forcast data ##########")
    # Create dict to define which data should be visualized as layers
    dict_figures = {
        "label": df_y.columns.values[1:],   # Skip the first column which includes the date (represents the x-axis)
    }
    plot_time_series_data_as_layers("baseline_model.html", "Baseline Model", "Forecast vs. Testdata", "sby_need", df_y.date, df_y[dict_figures.get('label')].columns.values, df_y[dict_figures.get('label')])

    # Metrics for Evaluation
    MSE, MAE, RMSE, MAPE, R2 = __baseline_model.evaluate(y_test.iloc[:,1:], y_hat)
    own_logger.info("########## Evaluation Metrics of the Baseline Model: ##########")
    own_logger.info("MSE = %f", MSE)
    own_logger.info("MAE = %f", MAE)
    own_logger.info("RMSE = %f", RMSE)
    own_logger.info("MAPE = %f", MAPE)
    own_logger.info("R2 = %f", R2)
