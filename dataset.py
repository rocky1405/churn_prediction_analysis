"""
    The file has functionality load the data and pre-process it by
    handling missing data, removing unwanted columns, handling multiple entries,
    convert numerical data into categorical data based on quantiles
    and label encoding the categorical columns
"""
import pandas as pd
import  numpy as np
from sklearn.preprocessing import LabelEncoder

__authors__ = "Radhakrishnan Iyer"
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Radhakrishnan Iyer"
__email__ = "srivatsan65@gmail.com"
__status__ = "Development"


class Dataset:
    """
        The class has functionality load the data and pre-process it by
        handling missing data, removing unwanted columns, handling multiple entries
        and label encoding the categorical columns
    """

    def __init__(self, data_path):
        """
            initialize all the variables
            :param data_path: the path of the input data
        """
        self.data = pd.read_csv(data_path)
        self.unwanted_columns = ["customerID", "PhoneService", "InternetService", "TotalCharges"]
        self.categorical_features_list = ["gender", "SeniorCitizen", "Partner", "Dependents",
                                          "MultipleLines", "Contract", "PaperlessBilling",
                                          "PaymentMethod", "Churn", "StreamingService"]

    def label_encoding_columns(self):
        """
            all the categorical columns containing text are label encoded into numbers
        """
        for column_name in self.categorical_features_list:
            label_encoder = LabelEncoder()
            self.data[column_name] = label_encoder.fit_transform(self.data[column_name])

    def handle_missing_values(self):
        """
            the missing values are handled by removing the rows since the
            number of rows with missing values is very less
        """
        self.data['TotalCharges'].replace(' ', np.nan, inplace=True)
        self.data = self.data.astype({"TotalCharges": "float64"})
        self.data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        return self.data

    def handle_column_entries(self):
        """
            the columns containing entries which would be better if
            changed to match with other entries or shorten them
            are handled
        """
        self.data["StreamingService"] = self.data["StreamingService"].replace(["No internet service"], "No")
        self.data["PaymentMethod"] = self.data["PaymentMethod"].replace(["Bank transfer (automatic)"], "Bank transfer")
        self.data["PaymentMethod"] = self.data["PaymentMethod"].replace(["Credit card (automatic)"], "Credit card")

    def drop_columns(self):
        """
            the unwanted columns which contain unique rows or the
            ones having high correlation with others are dropped
        """
        for column_names in self.unwanted_columns:
            self.data = self.data.drop(columns=[column_names])

    def pre_process_data(self):
        """
            all the functions defined to pre-process the data are called
        """
        self.handle_missing_values()
        self.handle_column_entries()
        self.drop_columns()
        self.label_encoding_columns()
        return self.data

    def establish_ranges(self, column_name, data_list, quantiles):
        """
            establishing the ranges based on the numerical data
            :param column_name: the name of the column to be processed
            :param data_list: the series from the dataset containing the required data
            :param quantiles: the number of quantiles we would like to split
        """
        range_4 = pd.qcut(data_list, q=quantiles, retbins=True)
        final_ranges = []
        for index, value in enumerate(range_4[1]):
            range_dict = {}
            if index != len(range_4[1])-1:
                range_dict["low_range"] = value
                range_dict["high_range"] = range_4[1][index + 1]
                final_ranges.append(range_dict)
        self.binning_data(column_name, data_list, final_ranges)

    def binning_data(self, column_name, data_list, range_list):
        """
            binning the value in each row into different categories
            obtained by establishing the ranges above
            :param column_name: the name of the column to be processed
            :param data_list: the series from the dataset containing the required data
            :param range_list: the different ranges based on quantiles
        """
        for range_dict in range_list:
            for value in data_list:
                if range_dict["low_range"] <= value < range_dict["high_range"]:
                    self.data[column_name + "Range"] = str(range_dict["low_range"]) + "-" + str(range_dict["high_range"])
        self.unwanted_columns.append(column_name)
