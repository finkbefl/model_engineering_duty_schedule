# Test the own_logging.py
#
# Start the test:
#   python -m tests.utils.own_logging_test

# Import the external packages
# Unittest Class for the inheritance
import unittest
# The base logging to compare
import logging

# Import internal packages/ classes
# The Class to test
from src.utils.own_logging import OwnLogging

class UnitTestOwnLogging(unittest.TestCase):
    """
    Unitttest class for the class OwnLogging from src.utils.own_logging
    """
    # Test to get the logger instance
    def test_getLogger(self):
        """
        Run Unittests th get the logger instance
        """
        # Without passing an package name
        result = OwnLogging().logger
        expectation = type(logging.getLogger())
        message = "given object is not instance of Logger."
        # Check if result is instance of class
        self.assertIsInstance(result, expectation, message)

        # With passing an correct package name
        result = OwnLogging("testpackage").logger
        expectation = type(logging.getLogger("testpackage"))
        message = "given object is not instance of Logger."
        # Check if result is instance of class
        self.assertIsInstance(result, expectation, message)

        # With passing an invalid type (int instead of str)
        with self.assertRaises(TypeError):
            result = OwnLogging(10).logger

# Call this script within the unittest context
if __name__ == '__main__':
    unittest.main()

