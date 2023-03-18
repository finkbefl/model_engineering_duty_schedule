# Script for data preparation

# Import the external packages
# Operating system functionalities
import sys
import os
from pathlib import Path
# Stream handling
import io
# To handle pandas data frames
import pandas as pd
# For various calculations
import numpy as np
# For preprocessing transformations
from sklearn.preprocessing import PowerTransformer
# Using statsmodel for detecting stationarity
from statsmodels.tsa.stattools import adfuller, kpss

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
                # Calculate the outlier boundaries
                lower_fence, upper_fence = calc_outlier_boundaries(y_datas[y_datas.columns[index]])
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
    __own_logger.info("Convert the column date to DateTime")
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

def calc_outlier_boundaries(df):
    """
    Function calculate the boundaries to detect outliers using the turkey method (boxplot)
    ----------
    Parameters:
        df : pandas.core.frame.DataFrame
            The data
    ----------
    Returns:
        lower_fence : float
            The calculated lower limit
        upper_fence : float
            The calculated upper limit
    """

    # Calculate boundaries using the tukey method
    q1, q3 = np.percentile(df, [25, 75])
    IRQ = q3 - q1
    lower_fence = q1 - (1.5 * IRQ)
    upper_fence = q3 + (1.5 * IRQ)
    __own_logger.info("Calculated outlier boundaries: lower_fence=%f, upper_fence=%f", lower_fence, upper_fence)

    return lower_fence, upper_fence

#########################################################

def stationarity_test(df):
    """
    Function to test the data columns for stationarity (Augmented Dickey-Fuller and Kwiatkowski-Phillips-Schmidt-Shin in combination for more confident decisions)
    ----------
    Parameters:
        df : pandas.core.frame.DataFrame
            The data
    ----------
    Returns:
        dict with the stationarity test results:
        {'column_name': 
            {'ADF': Boolean, 'KPSS': Boolean},
        ...
        }
    """

    stationarity_dict= {} # create an empty dictionary for the test results
    # Iterate over all columns except column one (date)
    for column in df.iloc[:,1:]:
        # Do not consider data which not vary over time, so skip the column which only consists of one value
        if (df[column].nunique() == 1):
            __own_logger.info("Skip column: %s because it not vary over time", column)
            continue
        # Check for stationarity
        # Augmented Dickey-Fuller Test
        adf_decision_stationary = False
        try:
            adf_output = adfuller(df[column])
            # Decision based on pval
            adf_pval = adf_output[1]
            if adf_pval < 0.05: 
                adf_decision_stationary = True
        except Exception as error:
            __own_logger.error("Error during ADF Test", exc_info=error)
        # Kwiatkowski-Phillips-Schmidt-Shin Test
        kpss_decision_stationary = False
        try:
            kpss_output = kpss(df[column])
            # Decision based on pval
            kpss_pval = kpss_output[1]
            if kpss_pval >= 0.05: 
                kpss_decision_stationary = True
        except Exception as error:
            __own_logger.error("Error during KPSS Test", exc_info=error)
        # Add the test results to the dict
        stationarity_dict[column] = {"ADF": adf_decision_stationary, "KPSS": kpss_decision_stationary}
    __own_logger.info("Stationarity: %s", stationarity_dict)

    return stationarity_dict

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
        The processed data as DataFrame:
        df_processed_data: Without consideration of stationarity
        df_stationary_data: Strict stationarity data
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
    # Check the datetime series to detect if a whole row is missing, so compare wit the date-range
    num_missing_rows = len(pd.date_range(start = df_processed_data.date.iloc[0], end = df_processed_data.date.iloc[-1]).difference(df_processed_data.date))
    __own_logger.info("Total number of missing rows: %d", num_missing_rows)
    # Handle missing rows
    if num_missing_rows == 0:
        __own_logger.info("No missing rows in the dataset, handling not necessary")
    else:
        __own_logger.warning("Missing rows in the dataset which must be preprocessed!")
        sys.exit('Fix missing rows in the dataset!')
        # TODO If there are missing rows in the dataset, these must be preprocessed!


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
    
    # Outliers
    # Detect skewed data for which the detection of outliers is difficult
    column_skew_values = df_processed_data.skew(axis='index')
    __own_logger.info("Skewness of the columns:\n%s",column_skew_values)
    # Skew values less than -1 or greater than 1 are highly skewed
    column_names_highly_skewed = set()
    for name,value in column_skew_values.items():
        if abs(value) > 1:
            column_names_highly_skewed.add(name)
    __own_logger.info("Columns which contains highly skewed data:\n%s",column_names_highly_skewed)
    #Transform highly skewed data for outlier detection
    df_transformed = pd.DataFrame()
    for column in column_names_highly_skewed:
        # reshape data to have rows and columns
        data = df_processed_data[column].values.reshape((len(df_processed_data[column].values),1))
        # Transform with boxcox, but data must be strictly positive, so use yeo-johnson which works with positive and negative values
        pt = PowerTransformer(method='yeo-johnson')
        df_transformed.insert(len(df_transformed.columns), column + "_trans", pd.DataFrame(pt.fit_transform(data)))
        # Add the transformed data columns to the existing DataFrame
        df_processed_data = pd.concat([df_processed_data, df_transformed], axis=1)
    # Visualize values out of the whiskers (borders of the boxplot)
    __own_logger.info("Visualize the whiskers to detect outliers")
    # Create dict to define which data should be visualized as figures
    dict_figures = {
        "label": ['n_sick', 'calls', 'sby_need'] + df_transformed.columns.values.tolist(),
        "title": ["Number of emergency drivers who have registered a sick call", 
                  "Number of emergency calls",
                  "Number of substitute drivers to be activated",
                  "Yeo-Johnson transformed number of substitute drivers to be activated"]
    }
    plot_time_series_data("data_prep_pot_outl.html", "Potential Outliers", dict_figures.get('title'), df_processed_data.date, df_processed_data[dict_figures.get('label')].columns.values, df_processed_data[dict_figures.get('label')], show_outliers=True)
    # Iterate over all columns except column 1 which contains the date and the columns which are highly skewed
    for column in df_processed_data.iloc[:,1:].columns.drop(column_names_highly_skewed):
        lower_fence, upper_fence = calc_outlier_boundaries(df_processed_data[column])
        # Get the row indizes which contains outliers
        row_indizes = df_processed_data[(df_processed_data[column] > upper_fence) | (df_processed_data[column] < lower_fence)].index
        # First trial: Drop the rows
        #__own_logger.info("Drop rows due to detected outliers in column %s: Number of outliers detected %d", column, len(row_indizes))
        #df_processed_data.drop(row_indizes, inplace=True)
        # Second trial: Replace the outlier with the value of the previous row
        __own_logger.info("Replace outliers in column %s: Number of outliers detected %d", column, len(row_indizes))
        # At first, set the outliers to NaN
        df_processed_data[column][row_indizes]=np.nan
        # Then fill the missing values (outliers) with the previos observation
        df_processed_data[column].ffill(inplace=True)
    # Drop the transformed data columns
    df_processed_data.drop(df_transformed.columns.values, inplace=True, axis=1)
    # TODO: Do not delete transformed data?

    # Time Series Stationarity
    # Copy the data for stationary data
    df_stationary_data = df_processed_data.copy()
    # Test the columns for stationarity
    stationarity_results = stationarity_test(df_stationary_data)
    # Iterate over the tested columns and apply a first order differencing
    for column, results in stationarity_results.items():
        # Get the results
        result_adf = results["ADF"]
        result_kpss = results["KPSS"]
        # Check for non-stationarity
        if result_adf == False or result_kpss == False:
            # non-stationary: Try a first order dfferencing
            __own_logger.info("The column %s is non-stationary, try a first order differencing", column)
            df_stationary_data[column] = df_stationary_data[column].diff()
            # Rename the column to add differencing information
            df_stationary_data.rename(columns={column:column + "_diff"}, inplace=True)
        else:
            # The time series is strict stationary
            __own_logger.info("The column %s is strict stationary", column)
    # The differenced data contains one less data point (NaN), which must be dropped
    df_stationary_data.dropna(inplace=True)
    # Test the columns again for stationarity
    stationarity_results = stationarity_test(df_stationary_data)
    # Are the columns now strict stationary?
    for column in stationarity_results:
        if stationarity_results[column].values() == False:
            __own_logger.error("########## Error during the data preparation for stationary data ##########")
            sys.exit('The data is not strict stationary! Fix it!')

    return df_processed_data, df_stationary_data

#########################################################
#########################################################
#########################################################

# When this script is called directly...
if __name__ == "__main__":
    # ...then calling the functions

    __own_logger.info("########## START ##########")

    # Loading the raw data as pandas DataFrame
    __own_logger.info("########## Loading the raw data ##########")
    df_raw_data = load_data("raw", "sickness_table.csv")

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
    df_processed_data, df_processed_stationary_data = data_preparation(df_preprocessed_data)

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
    # Visualize the prepared stationary data
    __own_logger.info("########## Visualize the prepared stationary data as time series ##########")
    # Create dict to define which data should be visualized as figures
    dict_figures = {
        "label": df_processed_stationary_data.columns.values[1:],   # Skip the first column which includes the date (represents the x-axis)
        "title": ["First order differencing of number of emergency drivers who have registered a sick call", 
                  "First order differencing of number of emergency calls",
                  "First order differencing of number of emergency drivers on duty",
                  "Number of available substitute drivers",
                  "First order differencing of number of substitute drivers to be activated"]
    }
    plot_time_series_data("prepared_input_data_stationary.html", "Prepared Stationary Input Data", dict_figures.get('title'), df_processed_stationary_data.date, df_processed_stationary_data[dict_figures.get('label')].columns.values, df_processed_stationary_data[dict_figures.get('label')])

    # Save the prepared data to csv file
    __own_logger.info("########## Save the prepared data as time series ##########")
    save_data(df_processed_data, "processed", "sickness_table_prepared.csv")
    # Save the prepared stationary data to csv file
    __own_logger.info("########## Save the prepared stationary data as time series ##########")
    save_data(df_processed_stationary_data, "processed", "sickness_table_prepared_stationary.csv")