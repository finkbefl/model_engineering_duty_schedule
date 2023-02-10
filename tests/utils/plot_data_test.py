# Test the plot_data.py
#
# Start the test:
#   python -m tests.utils.plot_data_test

# Import the external packages
# Unittest Class for the inheritance
import unittest
import os
# To create pandas series
import pandas as pd
import numpy as np
# Check types
from bokeh.plotting.figure import Figure

# Import internal packages/ classes
# The Class to test
from utils.plot_data import PlotBokeh, PlotMultipleLayers, PlotMultipleFigures
from utils.own_exceptions import IllegalArgumentError

class UnitTestPlotBokeh(unittest.TestCase):
    """
    Unitttest class for the class PlotBokeh, PlotMultipleLayers and PlotMultipleFigures from src.utils.plot_data
    """
    # Create pandas series data
    index =[1, 2, 3, 4, 5]
    data1 = np.array([10, 20, 30, 40, 50])
    data2 = np.array([100, 200, 300, 400, 500])
    series1 = pd.Series(data1, index)
    series2 = pd.Series(data2, index)
    # Test to create a PlotBokeh object
    def test_PlotBokeh(self):
        """
        Run Unittests to test the class PlotBokeh
        """
        # Correct Parameter
        file_name = os.path.join("5_output","unittestPlotBokeh.html")
        file_title = "Test Title"
        result = PlotBokeh(file_name, file_title)
        expectation = type(PlotBokeh(file_name, file_title))
        message = "given object is not instance of PlotBokeh."
        # Check if result is instance of class
        self.assertIsInstance(result, expectation, message)

        # Both parameter None is also allowed
        result = PlotBokeh()
        expectation = type(PlotBokeh())
        message = "given object is not instance of PlotBokeh."
        # Check if result is instance of class
        self.assertIsInstance(result, expectation, message)

        # Wrong Parameter: Only one Parameter as None is not allowed
        with self.assertRaises(IllegalArgumentError):
            PlotBokeh(file_name, None)
            PlotBokeh(None, file_title)

    # Test the PlotMultipleLayers with Circle Layer
    def test_PlotMultipleLayersCircle(self):
        """
        Run Unittests to test the class PlotMultipleLayers and add circle layers
        """
        # Correct Parameter (optional parameter tested within class test_PlotBokeh)
        file_name = os.path.join("5_output","unittestMultipleLayersCircle.html")
        file_title = "Test Title"
        PlotMultipleLayers("Test Title", "x", "y", file_name, file_title)

        # Wrong Parameter
        with self.assertRaises(TypeError):
            PlotMultipleLayers(1, "x", "y")
            PlotMultipleLayers("Test Title", PlotBokeh(), "y")
            PlotMultipleLayers("Test Title", "x", 2)
        with self.assertRaises(IllegalArgumentError):
            PlotMultipleLayers(None, "x", "y")
            PlotMultipleLayers("Test Title", "", None)

        # Add Circle layer
        figure = PlotMultipleLayers("Test Title", "x", "y")
        # Correct parameter
        figure.addCircleLayer("test", self.series1, self.series2)
        # Wrong parameter
        with self.assertRaises(TypeError):
            figure.addCircleLayer(1, self.series1, self.series2)
            figure.addCircleLayer("", self.series1, 2)
        with self.assertRaises(IllegalArgumentError):
            figure.addCircleLayer("", self.series1, self.series2)
            figure.addCircleLayer("test", None, self.series2)

        # Get figure
        result = figure.getFigure()
        expectation = Figure
        message = "given object is not instance of bokeh.plotting.figure.Figure."
        # Check if result is instance of class
        self.assertIsInstance(result, expectation, message)

        # Show Plot
        figure.showPlot()


    # Test the PlotMultipleLayers with vertical bar Layer
    def test_PlotMultipleLayersVBar(self):
        """
        Run Unittests to test the class PlotMultipleLayers and add vertival bar layers
        """
        file_name = os.path.join("5_output","unittestMultipleLayersVBar.html")
        file_title = "Test Title"
        figure2 = PlotMultipleLayers("Test Title 2", "x", "y", file_name, file_title)
        # Add vertical bar layer
        # Correct parameter
        figure2.addVBarLayer("test1", 1, 1)
        figure2.addVBarLayer("test2", 2, 3)
        figure2.addVBarLayer("test3", 3, 6)
        figure2.addVBarLayer("test4", 4, 9)
        figure2.addVBarLayer("test5", 5, 14)

        # Wrong Parameter
        with self.assertRaises(TypeError):
            figure2.addVBarLayer(2, 1, 1)
            figure2.addVBarLayer("test1", "test", 1)
        with self.assertRaises(IllegalArgumentError):
            figure2.addVBarLayer("test1", 1, None)

        # Show Plot
        figure2.showPlot()

    # Test the st axis range functionality
    def test_PlotMultipleLayersSetAxisRange(self):
        """
        Run Unittests to test the class PlotMultipleLayers and set axis ranges
        """
        file_name = os.path.join("5_output","unittestSetAxisRange.html")
        file_title = "Test Title"
        figure_3 = PlotMultipleLayers("Test Title", "x", "y", file_name, file_title)
        figure_3.addCircleLayer("test", self.series1, self.series2)

        # Correct parameter
        figure_3.set_axis_range(1,200)
        figure_3.set_axis_range(y_min=300, y_max=500)

        # Wrong parameter
        with self.assertRaises(TypeError):
            figure_3.set_axis_range("Test", 1, 2, 3)
            figure_3.set_axis_range(0, 1, 2, "test")
            figure_3.set_axis_range(0, None)
        with self.assertRaises(IllegalArgumentError):
            figure_3.set_axis_range(1)
            figure_3.set_axis_range(x_max=3)
            figure_3.set_axis_range(x_max=2, y_max=2)
            figure_3.set_axis_range(y_min=15)

    def test_PlotMultipleLayersAddGreenBox(self):
        """
        Run Unittests to test the class PlotMultipleLayers and add a green box
        """
        file_name = os.path.join("5_output","unittestAddBox.html")
        file_title = "Test Title"
        figure_4 = PlotMultipleLayers("Test Title", "x", "y", file_name, file_title)
        figure_4.addCircleLayer("test", self.series1, self.series2)

        # Correct parameter
        figure_4.add_green_box(300)

        # Wrong parameter
        with self.assertRaises(TypeError):
            figure_4.add_green_box("test")

        with self.assertRaises(IllegalArgumentError):
            figure_4.add_green_box(None)

        # Show Plot
        figure_4.showPlot()

    # Test the PlotMultipleFigures
    def test_PlotMultipleFigures(self):
        """
        Run Unittests to test the class PlotMultipleFigures
        """
        # Constructor parameter tested within class test_PlotBokeh
        file_name = os.path.join("5_output","unittestMultipleFigures.html")
        file_title = "Test Title"
        plot = PlotMultipleFigures(file_name, file_title)

        # Append figure
        # Correct parameter
        figure = PlotMultipleLayers("Test Title", "x", "y")
        figure.addCircleLayer("test", self.series1, self.series2)
        plot.appendFigure(figure.getFigure())
        figure2 = PlotMultipleLayers("Test Title 2", "x", "y", file_name, file_title)
        figure2.addVBarLayer("test1", 1, 1)
        figure2.addVBarLayer("test2", 2, 3)
        figure2.addVBarLayer("test3", 3, 6)
        figure2.addVBarLayer("test4", 4, 9)
        figure2.addVBarLayer("test5", 5, 14)
        plot.appendFigure(figure2.getFigure())

        # Wrong Parameter
        with self.assertRaises(TypeError):
            plot.appendFigure(file_name)
        with self.assertRaises(IllegalArgumentError):
            plot.appendFigure(None)

        # Show Plot
        plot.showPlot()


# Call this script within the unittest context
if __name__ == '__main__':
    unittest.main()