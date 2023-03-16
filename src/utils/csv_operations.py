# Functions for operations with csv files

# Import the external packages
# Operating system functionalities
import os
# To handle pandas data frames
import pandas as pd


def load_data(dirname, filename):
    """
    Loading the data from csv file to DataFrame
    ----------
    Parameters:
        dirname : String
            The name of the directory
        filename : String
            The name of the file
    ----------
    Returns:
        The data as pandas DataFrame
    """

    # Join the filepath of the raw data file
    filepath = os.path.join(os.path.dirname(__file__), "..", "..", "data", dirname, filename)

    # Read the data from CSV file
    data_raw = pd.read_csv(filepath)

    return data_raw

#########################################################

def save_data(df, dirname, filename):
    """
    Save the data into a csv file
    ----------
    Parameters:
        df : pandas.core.frame.DataFrame
            The data
        dirname : String
            The name of the directory
        filename : String
            The name of the file
    ----------
    Returns:
        no return
    """

    # Join the filepath of the raw data file
    filepath = os.path.join(os.path.dirname(__file__), "..", "..", "data", dirname, filename)

    # Save the data to CSV file
    df.to_csv(filepath, index=False)

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
    df.date = pd.to_datetime(df.date, errors='raise', utc=True)