# Script for data preparation

# Import the external packages
# Operating system functionalities
import sys
import os
# Stream handling
import io
# To handle pandas data frames
import pandas as pd
# For various calculations
import numpy as np

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

def plot_time_series_data(file_name, file_title, figure_titles, x_data, y_labels, y_datas, show_outliers=False):
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
        show_outliers : boolean
            Flag if the whiskers (borders of the boxplot) should be shown
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
            # Add the whiskers (borders of the boxplot) if the flag is enabled
            if show_outliers:
                # Calculate boundaries using the tukey method
                q1, q3 = np.percentile(y_datas[y_datas.columns[index]], [25, 75])
                IRQ = q3 - q1
                lower_fence = q1 - (1.5 * IRQ)
                upper_fence = q3 + (1.5 * IRQ)
                __own_logger.info("Add green box for calculated boundaries from %f to %f", lower_fence, upper_fence)
                figure.add_green_box(lower_fence, upper_fence)
            plot.appendFigure(figure.getFigure())
        # Show the plot in responsive layout, but only stretch the width
        plot.showPlotResponsive('stretch_width')
    except TypeError as error:
        __own_logger.error("########## Error when trying to plot data ##########", exc_info=error)
        sys.exit('A parameter does not match the given type')
    
#########################################################

def plot_scatter_data(file_name, file_title, figure_titles, x_labels, x_datas, y_labels, y_datas):
    """
    Function to plot scatter plots
    ----------
    Parameters:
        file_name : str
            The file name (html) in which the figure is shown
        file_title : str
            The output file title
        figure_titles : list
            The titles of the figures
        x_labels : array
            The label of the x axis
        x_datas : Series
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
        __own_logger.info("Plot scatter data with title %s as multiple figures to file %s", file_title, file_name)
        plot = PlotMultipleFigures(os.path.join("output",file_name), file_title)
        for (index, label) in enumerate(y_labels):
            __own_logger.info("Add figure for %s", label)
            figure = PlotMultipleLayers(figure_titles[index], x_labels[index], y_labels[index])
            figure.addCircleLayer(y_labels[index], x_datas.iloc[:,index], y_datas.iloc[:,index])
            plot.appendFigure(figure.getFigure())
        # Show the plot in fixed (not-responsive) layout
        plot.showPlotResponsive('fixed')
    except TypeError as error:
        __own_logger.error("########## Error when trying to plot data ##########", exc_info=error)
        sys.exit('A parameter does not match the given type')
    
#########################################################

def data_preprocessing(raw_data):
    """
    Function for initial data preprocessing
    ----------
    Parameters:
        raw_data : pandas.core.frame.DataFrame
            The raw data
    ----------
    Returns:
        The preorpcessed data as DataFrame
    """

    # Copy the data for preprocessing into a new variable
    __own_logger.info("Copy the DataFrame for preprocessing")
    df_preprocessed_data = raw_data.copy()
    # Convert the date
    __own_logger.info("Convert the date")
    # Convert the date
    convert_date_in_data_frame(df_preprocessed_data)

    # Drop the column with index 0 (unamed): Only contains the row number that is not needed (DataFrame has internal index)
    __own_logger.info("Drop the column with index 0 (unamed: row number)")
    df_preprocessed_data.drop(columns=df_preprocessed_data.columns[0], inplace=True)

    return df_preprocessed_data

#########################################################

def get_strong_correlated_columns(df, STRONG_CORR):
    """
    Function to detect strong correlated columns within a DataFrame
    ----------
    Parameters:
        df : pandas.core.frame.DataFrame
            The data
        STRONG_CORR : float
            The limit of the pearson correlation coefficient above which is strong correlation
    ----------
    Returns:
        col_strong_corr : set
            Set of all the column names with strong correlation to another (no duplicates)
        col_corr_related : list
            List of strong correlation related column names (duplicates possible)
    """

    # Correlation between the numeric variables to find the degree of linear relationship
    corr_matrix = df.corr(numeric_only=True, method='pearson')
    __own_logger.info("Pairwise correlation of columns: %s", corr_matrix)
    # Get variables with a strong correlation to another
    col_strong_corr = set()     # Set of all the column names with strong correlation to another
    col_corr_related = list()    # List of strong correlation related column names
    for i in range(len(corr_matrix.columns)):   # rows
        for j in range(i):                      # columns
            if (corr_matrix.iloc[i, j] >= STRONG_CORR) and (corr_matrix.columns[j] not in col_strong_corr):
                relatedname = corr_matrix.columns[i]        # getting the name of correlation matrix row
                colname = corr_matrix.columns[j]    # getting the name of the related column name
                __own_logger.info("Column name with strong (>= %f) correlation: %s (to %s)", STRONG_CORR, colname, relatedname)
                # TODO: Delete the colum from the dataset? But the correlation is not 1? Further analyzation necessary?
                # TODO: Print the scatter-plot between the two high correlated fetures,...
                # TODO Analyze the correlation with combined data: Combine sby_need and n_sby and correlate with dafted, then the correlation should be 1 and it can be dropped?
                # Add the columns to the set
                col_strong_corr.add(colname)
                col_corr_related.append(relatedname)
    __own_logger.info("Column names with strong correlation: %s", col_strong_corr)
    __own_logger.info("The strong correlation related column names: %s", col_corr_related)

    return col_strong_corr,col_corr_related

#########################################################

def data_preparation(preprocessed_data):
    """
    Function for data preparation
    ----------
    Parameters:
        raw_data : pandas.core.frame.DataFrame
            The preprocessed (raw) data
    ----------
    Returns:
        The processed data as DataFrame
    """

    # Copy the data for preparation into a new variable
    __own_logger.info("Copy the DataFrame for preparation")
    df_processed_data = preprocessed_data.copy()

    # Missing values
    # Detect missing values: In each columns
    __own_logger.info("Number of missing values in each column: %s", df_processed_data.isnull().sum())
    # Detect missing values: Total number
    missing_values_count = df_processed_data.isnull().sum().sum()
    __own_logger.info("Total number of missing values: %d", missing_values_count)
    # Handle missing values
    if missing_values_count == 0:
        __own_logger.info("No missing values in the dataset, handling not necessary")
    else:
        __own_logger.warning("Missing values in the dataset which must be preprocessed!")
        sys.exit('Fix missing values in the dataset!')
        # TODO If there are missing values in the dataset, these must be preprocessed!  


    # Redundancy: Rows
    # Get duplicated rows
    duplicate_rows_count = df_processed_data.duplicated().sum()
    __own_logger.info("Check duplicate rows and count the number: %d", duplicate_rows_count)
    # Handle duplicate_rows_count
    if duplicate_rows_count == 0:
        __own_logger.info("No duplicate rows in the dataset, handling not necessary")
    else:
        __own_logger.warning("Duplicate rows in the dataset which must be preprocessed!")
        sys.exit('Fix duplicate rows in the dataset!')
        # TODO If there are duplicate rows in the dataset, these must be preprocessed!
    
    # Redundancy: Columns
    # Discover columns that contain only a few different values
    __own_logger.info("Number of different values in each column: %s", df_processed_data.nunique())
    # Define with which pearson correlation coefficient a correlation is defined as strong
    STRONG_CORR = 0.9
    # Detect strong correlated columns (pearson correlation coefficient >= 0.9)
    col_strong_corr,col_corr_related = get_strong_correlated_columns(df_processed_data, STRONG_CORR)
    # Analyze relationship of correlated columns: Maybe in relation to n_sby?
    # dafted = sby_need - n_sby if positive, else zero? 
    # Add a new column to the dataframe at the end with the calculated differences and name it sby_need_calc'
    df_processed_data.insert(len(df_processed_data.columns),"sby_need_calc",df_processed_data.sby_need - df_processed_data.n_sby)
    # Remove all negative values
    df_processed_data.sby_need_calc[df_processed_data.sby_need_calc < 0] = 0
    __own_logger.info("sby_need_calc (sby_need - n_sby if positive, else zero): %s", df_processed_data.sby_need_calc)
    # Scatter plots of the high correlated columns and the relationship analysis
    x_labels = list(col_corr_related)
    x_labels.append("sby_need")
    x_labels.append("sby_need_calc")
    y_labels = list(col_strong_corr)
    y_labels.append("sby_need_calc")
    y_labels.append("dafted")
    dict_figures = {
        "x_label": x_labels,
        "y_label": y_labels,
        "title": ["Strong Correlation (>={}) between {} and {}".format(STRONG_CORR, y_label, x_label) for x_label,y_label in zip(x_labels,y_labels)]
    }
    plot_scatter_data("data_prep_high_corr_col.html", "High Correlated Columns", dict_figures.get('title'), df_processed_data[dict_figures.get('x_label')].columns.values, df_processed_data[dict_figures.get('x_label')], df_processed_data[dict_figures.get('y_label')].columns.values, df_processed_data[dict_figures.get('y_label')])
    # Detect exactly positive linear relationship (pearson correlation coefficient 1)
    col_strong_corr,col_corr_related = get_strong_correlated_columns(df_processed_data, 1)
    # Drop temporarly added column for correlation analysis reasons
    df_processed_data.drop(columns='sby_need_calc', inplace=True)
    # Drop column with exact positive linear leationship to another because of no additional information content
    df_processed_data.drop(columns=col_strong_corr, inplace=True)

    # Logging some information about the new DataFrame structure
    log_overview_data_frame(df_processed_data)
    
    # Outliers: Visualize values out of the whiskers (borders of the boxplot)
    __own_logger.info("Visualize the whiskers to detect outliers")
    # Create dict to define which data should be visualized as figures
    dict_figures = {
        "label": ['n_sick', 'calls', 'sby_need'],
        "title": ["Number of emergency drivers who have registered a sick call", 
                  "Number of emergency calls",
                  "Number of substitute drivers to be activated"]
    }
    plot_time_series_data("data_prep_pot_outl.html", "Potential Outliers", dict_figures.get('title'), df_processed_data.date, df_processed_data[dict_figures.get('label')].columns.values, df_processed_data[dict_figures.get('label')], show_outliers=True)
    # TODO: Without deeper domain knowledge it is hard to handle potential outliers (detect real outliers, remove, replace,...).
    #       The few values in n_sick and calls analyzed as outliers could be treated.
    #       For sb_need no statement can be made here whether it contains outliers.

    return df_processed_data



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

    # Preprocessing the data to prepare for a first visualization
    __own_logger.info("########## Preprocessing the data to prepare for a first visualization ##########")
    df_preprocessed_data = data_preprocessing(df_raw_data)

    # Logging some information about the preprocessed data
    __own_logger.info("########## Logging information about the preprocessed data ##########")
    log_overview_data_frame(df_preprocessed_data)

    # Visualize the preprocessed data as time series
    __own_logger.info("########## Visualize the preprocessed data as time series ##########")
    # Create dict to define which data should be visualized as figures
    dict_figures = {
        "label": df_preprocessed_data.columns.values[1:],   # Skip the first column which includes the date (represents the x-axis)
        "title": ["Number of emergency drivers who have registered a sick call", 
                  "Number of emergency calls",
                  "Number of emergency drivers on duty",
                  "Number of available substitute drivers",
                  "Number of substitute drivers to be activated",
                  "Number of additional duty drivers that have to be activated if the number of on-call drivers are not sufficient"]
    }
    plot_time_series_data("preprocessed_input_data.html", "Preprocessed (raw) Input Data", dict_figures.get('title'), df_preprocessed_data.date, df_preprocessed_data[dict_figures.get('label')].columns.values, df_preprocessed_data[dict_figures.get('label')])

    # Data preparation
    __own_logger.info("########## Preparing the data ##########")
    df_processed_data = data_preparation(df_preprocessed_data)

    # Logging some information about the prepared data
    __own_logger.info("########## Logging information about the prepared data ##########")
    log_overview_data_frame(df_processed_data)

    # Visualize the prepared data as time series
    __own_logger.info("########## Visualize the prepared data as time series ##########")
    # Create dict to define which data should be visualized as figures
    dict_figures = {
        "label": df_processed_data.columns.values[1:],   # Skip the first column which includes the date (represents the x-axis)
        "title": ["Number of emergency drivers who have registered a sick call", 
                  "Number of emergency calls",
                  "Number of emergency drivers on duty",
                  "Number of available substitute drivers",
                  "Number of substitute drivers to be activated"]
    }
    plot_time_series_data("prepared_input_data.html", "Prepared Input Data", dict_figures.get('title'), df_processed_data.date, df_processed_data[dict_figures.get('label')].columns.values, df_processed_data[dict_figures.get('label')])
