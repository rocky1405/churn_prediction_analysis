"""
    All the functionality related to exploratory data analysis for
    customer churn data set is implemented in this file
"""
import math
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from dataset import Dataset

__authors__ = "Radhakrishnan Iyer"
__copyright__ = ""
__credits__ = []
__license__ = ""
__version__ = "0.0.1"
__maintainer__ = "Radhakrishnan Iyer"
__email__ = "srivatsan65@gmail.com"
__status__ = "Development"


class DataAnalysis:
    """
        The class has functionality do exploratory data analysis on
        customer churn dataset
    """

    def __init__(self, data_path):
        """
            initialize all the variables
            :param data_path: the path of the folder containing dataset
        """
        self.data = Dataset(data_path).handle_missing_values()
        self.categorical_features_list = ["gender", "SeniorCitizen", "Partner", "Dependents", "MultipleLines",
                                          "Contract", "PaperlessBilling", "PaymentMethod", "Churn",
                                          "StreamingService", "PhoneService", "InternetService"]

    @staticmethod
    def conditional_entropy(x, y):
        """
            it is used as a sub function to find correlation between columns
            :param x: the column to find the correlation for
            :param y: the other columns with which correlation will be found
            :return: the entropy
        """
        y_counter = Counter(y)
        xy_counter = Counter(list(zip(x, y)))
        total_occurrences = sum(y_counter.values())
        entropy = 0
        for xy in xy_counter.keys():
            p_xy = xy_counter[xy] / total_occurrences
            p_y = y_counter[xy[1]] / total_occurrences
            entropy += p_xy * math.log(p_y / p_xy)
        return entropy

    def theil_u(self, x, y):
        """
            it is used as a sub function to find correlation between columns
            :param x: the column to find the correlation for
            :param y: the other columns with which correlation will be found
            :return: the correlation value between columns
        """
        s_xy = self.conditional_entropy(x, y)
        x_counter = Counter(x)
        total_occurrences = sum(x_counter.values())
        p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
        s_x = ss.entropy(p_x)
        if s_x == 0:
            return 1
        else:
            return (s_x - s_xy) / s_x

    def check_correlation(self):
        """
            it is used to find the correlation between one column and all the
            other columns for each and every column in the dataset
        """
        columns = self.data.columns
        for column_name in self.data:
            theilu = pd.DataFrame(index=[column_name], columns=self.data.columns)
            for index, value in enumerate(columns):
                u = self.theil_u(self.data[column_name].tolist(), self.data[value].tolist())
                theilu.loc[:, value] = u
            theilu.fillna(value=np.nan, inplace=True)
            plt.figure()
            sns.heatmap(theilu, annot=True, fmt='.2f')
            plt.show()

    def check_outliers(self):
        """
            the function will build a boxplot for numerical columns to find the
            outliers in the dataset
        """
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 7))
        fig.suptitle("Boxplot of 'Monthly Charges' and 'Total Charges'")
        boxprops = whiskerprops = capprops = medianprops = dict(linewidth=1)

        sns.boxplot(self.data['MonthlyCharges'], orient='v', color='#488ab5', ax=ax[0],
                    boxprops=boxprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops,
                    medianprops=medianprops)
        ax[0].set_facecolor('#f5f5f5')
        ax[0].set_yticks([20, 70, 120])

        sns.boxplot(self.data['TotalCharges'], orient='v', color='#488ab5', ax=ax[1],
                    boxprops=boxprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops,
                    medianprops=medianprops)
        ax[1].set_facecolor('#f5f5f5')
        ax[1].set_yticks([0, 4000, 8000])

        plt.tight_layout(pad=4.0)
        plt.show()

    def check_categorical_distribution(self):
        """
            this function will be a plot containing sub plots where
            the categorical data of the columns will be shown
        """
        rows, cols = 3, 4
        fig, ax = plt.subplots(rows, cols, figsize=(18, 18))
        row, col = 0, 0
        for i, categorical_feature in enumerate(self.categorical_features_list):
            if col == cols - 1:
                row += 1
            col = i % cols
            self.data[categorical_feature].value_counts().plot('bar', ax=ax[row, col], rot=0).set_title(categorical_feature)
        plt.show()

    def partner_vs_dependents(self):
        """
            this function will show the plot listing the relation
            between customers with or without partners having dependents
        """
        colors = ['#1c74b4', '#f9770d']
        partner_dependents = self.data.groupby(['Partner', 'Dependents']).size().unstack()

        ax = (partner_dependents.T * 100.0 / partner_dependents.T.sum()).T.plot(kind='bar',
                                                                                width=0.2,
                                                                                stacked=True,
                                                                                rot=0,
                                                                                figsize=(8, 6),
                                                                                color=colors)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(loc='center', prop={'size': 14}, title='Dependents', fontsize=14)
        ax.set_ylabel('% Customers', size=14)
        ax.set_title('% Customers with/without dependents based on whether they have a partner', size=14)
        ax.xaxis.label.set_size(14)

        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            ax.annotate('{:.0f}%'.format(height), (p.get_x() + .25 * width, p.get_y() + .4 * height),
                        color='white',
                        weight='bold',
                        size=14)
        plt.show()

    def tenure(self):
        """
            this function will build a plot listing the number customers
            in each tenure year with the service provider
        """
        ax = sns.distplot(self.data['tenure'], hist=True, kde=False,
                          bins=int(180 / 5), color='#1c74b4',
                          hist_kws={'edgecolor': 'black'},
                          kde_kws={'linewidth': 4})
        ax.set_ylabel('# of Customers')
        ax.set_xlabel('Tenure (months)')
        ax.set_title('# of Customers by their tenure')
        plt.show()

    def tenure_vs_contracts(self):
        """
            this function will show in plots the relation between each contract
            type and the tenure that each customer is with the service provider for
        """
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))

        ax = sns.distplot(self.data[self.data['Contract'] == 'Month-to-month']['tenure'],
                          hist=True, kde=False,
                          bins=int(180 / 5), color='#1c74b4',
                          hist_kws={'edgecolor': 'black'},
                          kde_kws={'linewidth': 4},
                          ax=ax1)
        ax.set_ylabel('# of Customers')
        ax.set_xlabel('Tenure (months)')
        ax.set_title('Month to Month Contract')

        ax = sns.distplot(self.data[self.data['Contract'] == 'One year']['tenure'],
                          hist=True, kde=False,
                          bins=int(180 / 5), color='#f9770d',
                          hist_kws={'edgecolor': 'black'},
                          kde_kws={'linewidth': 4},
                          ax=ax2)
        ax.set_xlabel('Tenure (months)', size=14)
        ax.set_title('One Year Contract', size=14)

        ax = sns.distplot(self.data[self.data['Contract'] == 'Two year']['tenure'],
                          hist=True, kde=False,
                          bins=int(180 / 5), color='#2ca42c',
                          hist_kws={'edgecolor': 'black'},
                          kde_kws={'linewidth': 4},
                          ax=ax3)

        ax.set_xlabel('Tenure (months)')
        ax.set_title('Two Year Contract')
        plt.show()

    def churn_vs_tenure(self):
        """
            the relation between churn and tenure is displayed in
            a boxplot
        """
        boxprops = whiskerprops = capprops = medianprops = dict(linewidth=1)
        sns.boxplot(x=self.data["Churn"], y=self.data["tenure"],
                    boxprops=boxprops,
                    whiskerprops=whiskerprops,
                    capprops=capprops,
                    medianprops=medianprops).set_title("Churn vs Tenure")
        plt.show()

    def churn_vs_contract(self):
        """
            the relation between the churn and different contract
            types are displayed in bar chart
        """
        colors = ['#1c74b4', '#f9770d']
        contract_churn = self.data.groupby(['Contract', 'Churn']).size().unstack()

        ax = (contract_churn.T * 100.0 / contract_churn.T.sum()).T.plot(kind='bar',
                                                                        width=0.3,
                                                                        stacked=True,
                                                                        rot=0,
                                                                        figsize=(10, 6),
                                                                        color=colors)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(loc='best', prop={'size': 14}, title='Churn')
        ax.set_ylabel('% Customers', size=14)
        ax.set_title('Churn by Contract Type', size=14)

        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            ax.annotate('{:.0f}%'.format(height), (p.get_x() + .25 * width, p.get_y() + .4 * height),
                        color='white',
                        weight='bold',
                        size=14)
        plt.show()

    def churn_vs_seniority(self):
        """
            the relation between churn and seniority level is displayed
            in bar charts
        """
        colors = ['#1c74b4', '#f9770d']
        seniority_churn = self.data.groupby(['SeniorCitizen', 'Churn']).size().unstack()

        ax = (seniority_churn.T * 100.0 / seniority_churn.T.sum()).T.plot(kind='bar',
                                                                          width=0.2,
                                                                          stacked=True,
                                                                          rot=0,
                                                                          figsize=(8, 6),
                                                                          color=colors)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(loc='center', prop={'size': 14}, title='Churn')
        ax.set_ylabel('% Customers')
        ax.set_title('Churn by Seniority Level', size=14)

        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            ax.annotate('{:.0f}%'.format(height), (p.get_x() + .25 * width, p.get_y() + .4 * height),
                        color='white',
                        weight='bold', size=14)
        plt.show()

    def perform_data_analysis(self):
        """
            all the functions performing exploratory data analysis
            will be called over here
        """
        self.check_outliers()
        self.check_correlation()
        self.check_categorical_distribution()
        self.partner_vs_dependents()
        self.tenure()
        self.tenure_vs_contracts()
        self.churn_vs_tenure()
        self.churn_vs_contract()
        self.churn_vs_seniority()
