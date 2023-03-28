# Skript for the model evaluation

# Import the external packages
# Operating system functionalities
import sys
import os
from pathlib import Path
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
            own_logger.info("Add Layer for %s", layer)
            figure.addLineCircleLayer(layer, x_data, y_datas[y_datas.columns[index]])
        return figure.getFigure()
    except TypeError as error:
        own_logger.error("########## Error when trying to create figure ##########", exc_info=error)
        sys.exit('A parameter does not match the given type')

#########################################################

def figure_vbar(figure_title, y_label, x_data, y_data):
    """
    Function to create a vbar chart figure with differet colors for every bar in the known color sequence
    ----------
    Parameters:
        figure_title : str
            The title of the figure
        y_label : str
            The label of the y axis
        x_data : numbers.Real
            The x data to plot
        y_data : numbers.Real
            The y data to plot
    ----------
    Returns:
        The figure
    """

    try:
        own_logger.info("Figure for vbar chart: %s", figure_title)
        figure = PlotMultipleLayers(figure_title, None, y_label, x_range=x_data)
        figure.addVBarLayer(x_data, y_data)
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

    # Set the forecast of the number of available substitude drivers to the forecast number of needed substitude drivers, if they are positive
    # TODO: Find a better "function" (e.g. only consider the forecast in some seasons?)
    df_y["n_sby_pred"] = y_pred.iloc[:,1:]
    # TODO: Add a offset to the predicted values?
    #df_y.n_sby_pred = df_y.n_sby_pred + 10
    df_y.n_sby_pred[df_y.n_sby_pred < 0] = 0
    # TODO: Add a offset to the data freed from negative values?
    #df_y.n_sby_pred = df_y.n_sby_pred + 20

    # Visualize the data
    # Add the constant variable manually as workaround: TODO: In future, the variable will vary over time and must be considered!
    df_y["n_sby"] = 90
    own_logger.info("########## Visualize the number of available substitude drivers ##########")
    # Create dict to define which data should be visualized as layers
    dict_figures = {
        "label": [df_y.n_sby.name, df_y.n_sby_pred.name],
    }
    figure_2 = figure_time_series_data_as_layers("Number of available substitude drivers", "n_sby", df_y.date, df_y[dict_figures.get('label')].columns.values, df_y[dict_figures.get('label')])

    # Too less available substitude drivers: Positive differences between the number of needed and available substitude drivers
    # Testdata n_sby
    df_y["n_sby_diff_pos"] = y_test.sby_need - 90
    # Forecast n_sby
    df_y["n_sby_pred_diff_pos"] = y_test.sby_need - df_y["n_sby_pred"]
    # Remove all negative values
    df_y.n_sby_diff_pos[df_y.n_sby_diff_pos < 0] = 0
    df_y.n_sby_pred_diff_pos[df_y.n_sby_pred_diff_pos < 0] = 0

    # Visualize the data
    own_logger.info("########## Visualize the too less available substitude drivers ##########")
    # Create dict to define which data should be visualized as layers
    dict_figures = {
        "label": [df_y.n_sby_diff_pos.name, df_y.n_sby_pred_diff_pos.name],
    }
    figure_3 = figure_time_series_data_as_layers("Too less available substitude drivers", "n_sby_diff_pos", df_y.date, df_y[dict_figures.get('label')].columns.values, df_y[dict_figures.get('label')])
    
    # Percentage of days with enough available substitude drivers?
    # Create dict for the percentage of days with enough available substitude drivers
    dict_enough_precentage = {
        "label": ['enough_percentage', 'enough_percentage_pred'],
        "value": [(df_y.n_sby_diff_pos == 0).sum()/len(df_y.n_sby_diff_pos), (df_y.n_sby_pred_diff_pos == 0).sum()/len(df_y.n_sby_pred_diff_pos)],
    }
    own_logger.info("Percentage of days with enough available substitude drivers: Testdata: %f, Prediction: %f", dict_enough_precentage.get('value')[0], dict_enough_precentage.get('value')[1])
    # Show Bar-Chart for the mean percentage of the activated substitude drivers
    figure_4 = figure_vbar("Percentage of days with enough available substitude drivers", "enough_percentage", dict_enough_precentage.get('label'), dict_enough_precentage.get('value'))

    # Percentage of activated substitude drivers (Replace division by zero with 0 as result and the maximum value is limited to 1 which is 100%)
    # Testdata
    df_y["need_percentage"] = df_y.sby_need.div(df_y.n_sby).replace(np.inf, 0)
    df_y.need_percentage[df_y.need_percentage > 1] = 1
    # Forecast
    df_y["need_percentage_pred"] = df_y.sby_need.div(df_y.n_sby_pred).replace(np.inf, 0)
    df_y.need_percentage_pred[df_y.need_percentage_pred > 1] = 1

    # Visualize the data
    own_logger.info("########## Visualize the percentage of the activated substitude drivers ##########")
    # Create dict to define which data should be visualized as layers
    dict_figures = {
        "label": [df_y.need_percentage.name, df_y.need_percentage_pred.name],
    }
    figure_5 = figure_time_series_data_as_layers("Percentage of the activated substitude drivers", "need_percentage", df_y.date, df_y[dict_figures.get('label')].columns.values, df_y[dict_figures.get('label')])
    # Create dict for the mean percentage of the activated substitude drivers
    dict_need_precentage = {
        "label": [df_y.need_percentage.name, df_y.need_percentage_pred.name],
        "value": [df_y.need_percentage.mean(), df_y.need_percentage_pred.mean()],
    }
    own_logger.info("Mean percentage of the activated substitude drivers: Testdata: %f, Prediction: %f", dict_need_precentage.get('value')[0], dict_need_precentage.get('value')[1])
    # Show Bar-Chart for the mean percentage of the activated substitude drivers
    figure_6 = figure_vbar("Mean percentage of the activated substitude drivers", "need_percentage", dict_need_precentage.get('label'), dict_need_precentage.get('value'))

    # Create the plot with the created figures
    file_name = "model_evaluation.html"
    file_title = "Model-Evaluation"
    own_logger.info("Plot times series data with title %s as multiple figures to file %s", file_title, file_name)
    plot = PlotMultipleFigures(os.path.join("output",file_name), file_title)
    plot.appendFigure(figure_1)
    plot.appendFigure(figure_2)
    plot.appendFigure(figure_3)
    plot.appendFigure(figure_4)
    plot.appendFigure(figure_5)
    plot.appendFigure(figure_6)
    # Show the plot in responsive layout, but only stretch the width
    plot.showPlotResponsive('stretch_width')