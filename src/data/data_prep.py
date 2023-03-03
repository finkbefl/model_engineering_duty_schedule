# Import the external packages
# Operating system functionalities
import sys
import os
# Stream handling
import io
# To handle pandas data frames
import pandas as pd

# Import internal packages/ classes
# Import the src-path to sys path that the internal modules can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
# To handle the Logging for all modules in the same way
from utils.own_logging import OwnLogging
# To plot data with bokeh
from utils.plot_data import PlotMultipleLayers, PlotMultipleFigures

#########################################################

# Initialize the logger
__own_logger = OwnLogging(__name__).logger

#########################################################

def load_raw_data():
    """
    Loading the raw data
    ----------
    Parameters:
        no parameters
    ----------
    Returns:
        The raw data as pandas DataFrame
    """

    # Join the filepath of the raw data file
    filepath = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "sickness_table.csv")
    __own_logger.info("Filepath: %s", filepath)

    # Read the data from CSV file
    data_raw = pd.read_csv(filepath)

    return data_raw

#########################################################

def log_overview_data_frame(df):
    """
    Logging some information about the data frame
    ----------
    Parameters:
        df : pandas.core.frame.DataFrame
            The data
    ----------
    Returns:
        no returns
    """

    # Print the first 5 rows
    __own_logger.info("Data Structure (first 5 rows): %s", df.head(5))
    # Print some information (pipe output of DataFrame.info to buffer instead of sys.stdout for correct logging)
    buffer = io.StringIO()
    buffer.write("Data Info: ")
    df.info(buf=buffer)
    __own_logger.info(buffer.getvalue())

#########################################################

def convert_date_in_data_frame(df):
    """
    Function to convert the date object into DateTime
    ----------
    Parameters:
        df : pandas.core.frame.DataFrame
            The data
    ----------
    Returns:
        no returns
    """

    # Convert the date objects into DateTime (raise an exception when parsing is invalid)
    __own_logger.info("Convert the column date to DateTime")
    df.date = df.date.apply(lambda x: pd.to_datetime(x, errors='raise', utc=True))

#########################################################

def plot_time_series_data(file_name, file_title, figure_titles, x_data, y_labels, y_datas):
    """
    Function to plot time series data
    ----------
    Parameters:
        file_name : str
            The file name (html) in which the figure is shown
        file_title : str
            The output file title
        figure_titles : list
            The titles of the figures
        x_data : Series
                The x data to plot
        y_labels : array
            The label of the y axis
        y_datas : DataFrame
                The y data to plot
    ----------
    Returns:
        no returns
    """

    try:
        plot = PlotMultipleFigures(os.path.join("output",file_name), file_title)
        for (index, column) in enumerate(y_labels):
            figure = PlotMultipleLayers(figure_titles[index], "date", y_labels[index], x_axis_type='datetime')
            figure.addCircleLayer(y_labels[index], x_data, y_datas[y_datas.columns[index]])
            plot.appendFigure(figure.getFigure())
        plot.showPlot()
    except TypeError as error:
        __own_logger.error("########## Error when trying to plot data ##########", exc_info=error)
        sys.exit('A parameter does not match the given type')
    




#########################################################
#########################################################
#########################################################

# When this script is called directly...
if __name__ == "__main__":
    # ...then calling the functions

    __own_logger.info("########## START ##########")

    # Loading the raw data as pandas DataFrame
    __own_logger.info("########## Loading the raw data ##########")
    df_raw_data = load_raw_data()

    # Logging some information about the raw data
    __own_logger.info("########## Logging information about the raw data ##########")
    log_overview_data_frame(df_raw_data)

    # Convert the date
    __own_logger.info("########## Convert the date ##########")
    # Copy the data for conversion into a new variable
    df_converted_data = df_raw_data.copy()
    convert_date_in_data_frame(df_converted_data)

    # Drop the column with index 0 (unamed)
    __own_logger.info("########## Drop the column with index 0 ##########")
    df_converted_data.drop(columns=df_converted_data.columns[0], inplace=True)

    # Logging some information about the converted data
    __own_logger.info("########## Logging information about the converted data ##########")
    log_overview_data_frame(df_converted_data)

    # Visualize the raw data
    __own_logger.info("########## Visualize the raw data ##########")
    # Create dict to define which data should be visualized
    dict_to_visualize = {
        "label": df_converted_data.columns.values[1:],
        "title": ["Number of emergency drivers who have registered a sick call", 
                  "Number of emergency calls",
                  "Number of emergency drivers on duty",
                  "Number of available substitute drivers",
                  "Number of substitute drivers to be activated",
                  "Number of additional duty drivers that have to be activated if the number of on-call drivers are not sufficient"]
    }
    plot_time_series_data("raw_input_data.html", "Raw Input Data", dict_to_visualize.get('title'), df_converted_data.date, df_converted_data[dict_to_visualize.get('label')].columns.values, df_converted_data[dict_to_visualize.get('label')])