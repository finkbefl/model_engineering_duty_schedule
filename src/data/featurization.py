# Script for data featurization and data split

# Import the external packages
# Operating system functionalities
import sys
import os
from pathlib import Path
# For splitting of the dataset
from sklearn.model_selection import train_test_split

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
__own_logger = OwnLogging(Path(__file__).stem).logger

#########################################################

def data_feature_selection(df):
    """
    Function for selecting the features
    ----------
    Parameters:
        df : pandas.core.frame.DataFrame
            The data
    ----------
    Returns:
        The featurized data as DataFrame
    """

    # Using all prepared columns
    __own_logger.info("Select all prepared columns: %s", df.columns)
    df_featurized = df_prepared.copy()

    return df_featurized

#########################################################

def data_split(df, test_size):
    """
    Function for splitting the data into train and test set
    ----------
    Parameters:
        df : pandas.core.frame.DataFrame
            The data to split
        test_size : float
            The test size as proportion of the whole dataset (value between 0 and 1)
    ----------
    Returns:
        The splitted data as DataFrame:
        df_train: Without consideration of stationarity
        df_test: Strict stationarity data
    """

    __own_logger.info("Split the data with a test size of %f", test_size)
    df_train, df_test = train_test_split(df, test_size=test_size, shuffle=False)

    return df_train, df_test

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
        __own_logger.info("Plot times series data with title %s as multiple figures to file %s", file_title, file_name)
        plot = PlotMultipleFigures(os.path.join("output",file_name), file_title)
        for (index, label) in enumerate(y_labels):
            __own_logger.info("Add figure for %s", label)
            figure = PlotMultipleLayers(figure_titles[index], "date", y_labels[index], x_axis_type='datetime')
            figure.addCircleLayer(y_labels[index], x_data, y_datas[y_datas.columns[index]])
            plot.appendFigure(figure.getFigure())
        # Show the plot in responsive layout, but only stretch the width
        plot.showPlotResponsive('stretch_width')
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

    # Loading the prepared data as pandas DataFrame
    __own_logger.info("########## Loading the prepared data ##########")
    df_prepared = load_data("processed", "sickness_table_prepared.csv")
    df_prepared_stationary = load_data("processed", "sickness_table_prepared_stationary.csv")
    # Convert the date
    __own_logger.info("########## Convert the column date to DateTime ##########")
    convert_date_in_data_frame(df_prepared)
    convert_date_in_data_frame(df_prepared_stationary)

    # Featurize the data: Non-stationary for the time being
    __own_logger.info("########## Featurize the data ##########")
    # TODO: Using the stationary data?
    df_featurized = data_feature_selection(df_prepared)

    # Split the data into train and test set: Testset should contain a whole seasonal period, e.g. 1 from 3 years
    __own_logger.info("########## Split the data into train and test set ##########")
    df_train, df_test = data_split(df_featurized, 1/3)

    # Visualize the train data
    __own_logger.info("########## Visualize the train data as time series ##########")
    # Create dict to define which data should be visualized as figures
    dict_figures = {
        "label": df_train.columns.values[1:],   # Skip the first column which includes the date (represents the x-axis)
        "title": ["Number of emergency drivers who have registered a sick call", 
                  "Number of emergency calls",
                  "Number of emergency drivers on duty",
                  "Number of available substitute drivers",
                  "Number of substitute drivers to be activated"]
    }
    plot_time_series_data("train_data.html", "Train Data", dict_figures.get('title'), df_train.date, df_train[dict_figures.get('label')].columns.values, df_train[dict_figures.get('label')])

    # Visualize the test data
    __own_logger.info("########## Visualize the test data as time series ##########")
    # Create dict to define which data should be visualized as figures
    dict_figures = {
        "label": df_test.columns.values[1:],   # Skip the first column which includes the date (represents the x-axis)
        "title": ["Number of emergency drivers who have registered a sick call", 
                  "Number of emergency calls",
                  "Number of emergency drivers on duty",
                  "Number of available substitute drivers",
                  "Number of substitute drivers to be activated"]
    }
    plot_time_series_data("test_data.html", "Test Data", dict_figures.get('title'), df_test.date, df_test[dict_figures.get('label')].columns.values, df_test[dict_figures.get('label')])
    
    # Save the train data to csv file
    __own_logger.info("########## Save the train data as time series ##########")
    save_data(df_train, "modeling", "train_data.csv")
    # Save the test data to csv file
    __own_logger.info("########## Save the test data as time series ##########")
    save_data(df_test, "modeling", "test_data.csv")




