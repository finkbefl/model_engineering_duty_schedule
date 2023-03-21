# Skript for the model evaluation

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
            own_logger.info("Add Layer for %s", layer)
            figure.addLineCircleLayer(layer, x_data, y_datas[y_datas.columns[index]])
        return figure.getFigure()
    except TypeError as error:
        own_logger.error("########## Error when trying to create figure ##########", exc_info=error)
        sys.exit('A parameter does not match the given type')

#########################################################
#########################################################
#########################################################

# When this script is called directly...
if __name__ == "__main__":
    # ...then calling the functions

    own_logger.info("########## START ##########")

    # Loading the featurized data as pandas DataFrame
    own_logger.info("########## Loading the data ##########")
    y_test = load_data("modeling", "test_target.csv")
    y_pred = load_data("modeling", "y_pred.csv")

    # Convert the date
    own_logger.info("########## Convert the column date to DateTime ##########")
    convert_date_in_data_frame(y_test)
    convert_date_in_data_frame(y_pred)

    # Visualize the data
    # Merge the y_test with the y_pred data
    df_y = y_test.copy()
    df_y[y_test.sby_need.name + "_pred"] = y_pred.iloc[:,1:]
    own_logger.info("########## Visualize the test- and forecast-data ##########")
    # Create dict to define which data should be visualized as layers
    dict_figures = {
        "label": df_y.columns.values[1:],   # Skip the first column which includes the date (represents the x-axis)
    }
    figure_1 = figure_time_series_data_as_layers("SARIMAX-Model: Forecast vs. Testdata", "sby_need", df_y.date, df_y[dict_figures.get('label')].columns.values, df_y[dict_figures.get('label')])

    # Set the forecast of the number of available substitude drivers to the forecast number of needed substitude drivers
    # TODO: Find a better "function"?
    df_y["n_sby_pred"] = y_pred.iloc[:,1:]

    # days with too less available substitude drivers?
    # Testdata n_sby
    df_y["n_sby_less"] = y_test.sby_need - 90
    # Forecast n_sby
    df_y["n_sby_pred_less"] = y_test.sby_need - df_y["n_sby_pred"]
    # Remove all negative values
    df_y.n_sby_less[df_y.n_sby_less < 0] = 0
    df_y.n_sby_pred_less[df_y.n_sby_pred_less < 0] = 0
    # Count number of not zeros to get the number of days
    own_logger.info("Days with too less substitude drivers: Testdata: %d, Prediction: %d", (df_y.n_sby_less != 0).sum(), (df_y.n_sby_pred_less != 0).sum())

    # Visualize the data
    own_logger.info("########## Visualize the days with too less substitude drivers ##########")
    # Create dict to define which data should be visualized as layers
    dict_figures = {
        "label": [df_y.n_sby_less.name, df_y.n_sby_pred_less.name],   # Skip the first column which includes the date (represents the x-axis)
    }
    figure_2 = figure_time_series_data_as_layers("Days with too less substitude drivers", "n_sby_less", df_y.date, df_y[dict_figures.get('label')].columns.values, df_y[dict_figures.get('label')])

    # Create the plot with the created figures
    file_name = "model_evaluation.html"
    file_title = "Model-Evaluation"
    own_logger.info("Plot times series data with title %s as multiple figures to file %s", file_title, file_name)
    plot = PlotMultipleFigures(os.path.join("output",file_name), file_title)
    plot.appendFigure(figure_1)
    plot.appendFigure(figure_2)
    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')