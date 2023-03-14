# Skript for training a baseline model (Any model that is worse than this is not worth being considered at all)

# Import the external packages
# Operating system functionalities
import sys
import os
from pathlib import Path

# Import internal packages/ classes
# Import the src-path to sys path that the internal modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
# To handle the Logging for all modules in the same way
from utils.own_logging import OwnLogging
# To plot data with bokeh
from utils.plot_data import PlotMultipleLayers, PlotMultipleFigures
# To handle csv files
from utils.csv_operations import load_data, save_data, convert_date_in_data_frame
# For ExponentialSmoothing Forecasting
from statsmodels.tsa.api import ExponentialSmoothing

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

class BaselineModel():
    """
    A class for a baseline model
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
        own_logger.info("Initialize a baseline model")
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
        # Train the baseline model
        own_logger.info("Train the baseline model")
        # TODO: Seasonal period should be detected automatically!
        #exp = ExponentialSmoothing(y_train, trend='additive', seasonal='additive')
        exp = ExponentialSmoothing(y_train.iloc[:,1:], trend='additive', seasonal='additive', seasonal_periods=365)
        self.__model = exp.fit(use_brute=True, optimized=True)

    def predict(self, len):
        # Prediction
        own_logger.info("Prediction of %d values", len)
        y_hat = self.__model.forecast(len)
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

    # Train the baseline model
    own_logger.info("########## Train the baseline model ##########")
    __baseline_model = BaselineModel()
    __baseline_model.train(X_train, y_train)

    # Forecast
    own_logger.info("########## Forecasting ##########")
    y_hat = __baseline_model.predict(len(y_test))

    # Visualize the forecast data
    # Merge the y_test with the y_hat data
    df_y = y_test.copy()
    df_y['y_hat'] = y_hat.values
    own_logger.info("########## Visualize the forcast data ##########")
    # Create dict to define which data should be visualized as layers
    dict_figures = {
        "label": df_y.columns.values[1:],   # Skip the first column which includes the date (represents the x-axis)
    }
    plot_time_series_data_as_layers("baseline_model.html", "Baseline Model", "Forecast vs. Testdata", "sby_need", df_y.date, df_y[dict_figures.get('label')].columns.values, df_y[dict_figures.get('label')])
    
