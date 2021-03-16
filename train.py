import sys
from trainer import Trainer
from data_analysis import DataAnalysis

input = sys.argv[1]
model_name = sys.argv[2]

def analyze_and_train():
    # to view the different metrics obtained from the data run perform data analysis
    eda = DataAnalysis(input)
    eda.perform_data_analysis()
    # to train a model run the trainer function
    tr = Trainer(data_path=input, model_name=model_name)
    tr.train()
    tr.eval()
    tr.save()

if __name__ == "__main__":
    analyze_and_train()