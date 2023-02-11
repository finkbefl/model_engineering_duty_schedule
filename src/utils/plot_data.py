# Class to plot data with bokeh

# Import the external packages
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.layouts import gridplot
from bokeh.palettes import Category10_10
from pandas.core.series import Series
from bokeh.plotting.figure import Figure
from bokeh.models import Range1d, BoxAnnotation
from numbers import Real
# Import internal packages/ classes
from utils.own_logging import OwnLogging
from utils.check_parameter import checkParameter, checkParameterString
from utils.own_exceptions import IllegalArgumentError

class PlotBokeh():
    """
    A basic class for easy plotting with bokeh and output the result in a static HTML file if required
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

    # Initialize the logger
    __own_logging = OwnLogging(__name__)
    __own_logger = __own_logging.logger

    # Constructor Method
    def __init__(self, file_name=None, file_title=None):
        # Output to static HTML file
        if file_name is not None and file_title is not None:
            checkParameterString(file_name)
            checkParameterString(file_title)
            output_file(file_name, title=file_title)
            self.__own_logger.info("Bokeh plot initialized for output file %s", file_name)
        elif file_name is None and file_title is None:
            # No output file, used when a inherited class is used to fill a complex visualization file --> A valid state!
            pass
        else:
            raise IllegalArgumentError("A parameter cannot be None")

class PlotMultipleLayers(PlotBokeh):
    """
    A class for easy plotting multiple layers with bokeh and output the result in a static HTML file if required
    ----------
    Attributes:
        figure_title : str
            The title of the figure
        x_label : str
            The label of the x axis
        y_label : str
            The label of the y axis
        file_name : str
            The file name (html) in which the figure is shown
        file_title : str
            The output file title
    ----------
    Methods
        addCircleLayer(legend_label, x_data, y_data):
            Add a circle layer to the figure with the given data
        addVBarLayer(legend_label, x_data, y_data):
            Add a vertical bar layer to the figure with the given data
        add_green_box(top_val):
            Add a green box from y=0 up to the specified y value
        getFigure():
            Get the figure
        showPlot():
            show the figure
    """

    # Initialize the logger
    __own_logging = OwnLogging(__name__)
    __own_logger = __own_logging.logger

    # Constructor Method
    def __init__(self, figure_title, x_label, y_label, file_name=None, file_title=None):
        # Call the Base Class Constructor
        PlotBokeh.__init__(self, file_name, file_title)
        # For color cycling (different colors for the different layers)
        self.__color_iter = Category10_10.__iter__()
        # create a figure, but first check the parameter
        checkParameterString(figure_title)
        checkParameterString(x_label)
        checkParameterString(y_label)
        self.__own_figure = figure(title=figure_title, x_axis_label=x_label, y_axis_label=y_label)
        self.__own_logger.info("Bokeh plot for multiple layers initialized for figure %s", figure_title)

    def addCircleLayer(self, legend_label, x_data, y_data):
        """
        Add a layer to the figure (circle representation)
        ----------
        Parameters:
            legend_label : str
                The legend label of the layer
            x_data : Series
                The x data to plot
            y_data : Series
                The y data to plot
        ----------
        Returns:
            no returns
        """
        # check the parameter
        checkParameterString(legend_label)
        checkParameter(x_data, Series)
        checkParameter(y_data, Series)
        # add a plot to the figure, assign the next color automatically from the color iterator
        self.__own_figure.circle(x=x_data, y=y_data, legend_label=legend_label, color=next(self.__color_iter))
        self.__own_logger.info("Added  circle layer %s", legend_label)

    def addVBarLayer(self, legend_label, x_data, y_data):
        """
        Add a layer to the figure (vertical bar representation)
        ----------
        Parameters:
            legend_label : str
                The legend label of the layer
            x_data : numbers.Real
                The x data to plot
            y_data : numbers.Real
                The y data to plot
        ----------
        Returns:
            no returns
        """
        # check the parameter
        checkParameterString(legend_label)
        checkParameter(x_data, Real)
        checkParameter(y_data, Real)
        # add a plot to the figure, assign the next color automatically from the color iterator
        self.__own_figure.vbar(x = x_data, top = y_data, legend_label=legend_label, width=0.8, color=next(self.__color_iter))
        self.__own_logger.info("Added vbar layer %s", legend_label)

    def set_axis_range(self, x_min=None, x_max=None, y_min=None, y_max=None):
        """
        Set the axis range of the figure
        ----------
        Parameters:
            x_min : numbers.Real
                The min x value
            x_max : numbers.Real
                The max x value
            y_min : numbers.Real
                The min y value
            y_max : numbers.Real
                The max y value
        ----------
        Returns:
            no returns
        """
        checkParameter(x_min, Real, True)
        checkParameter(x_max, Real, True)
        checkParameter(y_min, Real, True)
        checkParameter(y_max, Real, True)
        if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
            self.__own_figure.x_range = Range1d(x_min, x_max)
            self.__own_figure.y_range = Range1d(y_min, y_max)
        elif x_min is not None and x_max is not None and y_min is None and y_max is None:
            self.__own_figure.x_range = Range1d(x_min, x_max)
        elif y_min is not None and y_max is not None and x_min is None and x_max is None:
            self.__own_figure.y_range = Range1d(y_min, y_max)
        else:
            raise IllegalArgumentError("A range must be a value pair!")

    def add_green_box(self, top_val):
        """
        Add a green box from y=0 up to the specified y value
        ----------
        Parameters:
            top_val : numbers.Real
                The upper y value as limit for the color box
        ----------
        Returns:
            no returns
        """
        checkParameter(top_val, Real, False)
        green_box = BoxAnnotation(top=top_val, bottom=0, fill_alpha=0.1, fill_color="green")
        self.__own_figure.add_layout(green_box)


    def getFigure(self):
        """
        Get the figure
        ----------
        Parameters:
            no parameter
        ----------
        Returns:
            Returns the figure (bokeh.plotting.figure.Figure)
        """
        return self.__own_figure

    def showPlot(self):
        """
        Show the plot
        ----------
        Parameters:
            no parameter
        ----------
        Returns:
            Show the plot
        """
        # show the figure
        self.__own_logger.info("Show the plot")
        show(self.__own_figure)

class PlotMultipleFigures(PlotBokeh):
    """
    A class for easy plotting multiple figures with bokeh and output the result in a static HTML file
    ----------
    Attributes:
        file_name : str
            The file name (html) in which the figure is shown
        file_title : str
            The output file title
    ----------
    Methods
        appendFigure(figure):
            Add a figure (bokeh.plotting.figure.Figure) to the plot
        showPlot():
            show the plot
    """

    # Initialize the logger
    __own_logging = OwnLogging(__name__)
    __own_logger = __own_logging.logger

    # Constructor Method
    def __init__(self, file_name, file_title):
        # Call the Base Class Constructor
        PlotBokeh.__init__(self, file_name, file_title)
        # The list of figures, which will be plotted
        self.__figure_list = []
        self.__own_logger.info("Bokeh plot for multiple figures initialized for file name %s", file_name)

    def appendFigure(self, figure):
        """
        Add a fiigure to the plot
        ----------
        Parameters:
            figure : bokeh.plotting.figure.Figure
                The figure to add
        ----------
        Returns:
            no returns
        """
        # check the parameter
        checkParameter(figure, Figure)
        self.__figure_list.append(figure)
        self.__own_logger.info("Appendend figure to Bokeh plot %s", figure)

    def showPlot(self):
        """
        Show the plot
        ----------
        Parameters:
            no parameter
        ----------
        Returns:
            Show the plot
        """
        # Add the figures out of the list to the gridplot (with 2 colums) and show it
        self.__own_logger.info("Show the gridplot")
        plot = gridplot(self.__figure_list, ncols=2)
        show(plot)
