from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd 

def main():
    print("Hello world")
    load_data()

def load_data():
    dataset = "exams.csv"
    path = "data/" + dataset

    dataset_names = ["gender","ethnicity","parental_education","lunch","test_prep_course","math_score","reading_score","writing_score"]

    df = pd.read_csv(path, names=dataset_names) 
    prepare_data(df)

def prepare_data(df):
    df.drop(index=df.index[0], axis=0, inplace=True)
    #print(df)
    for i in df.columns:     
        #print(i)
        try:
            df[i] = pd.to_numeric(df[i])
        except Exception as e:
            #print(e)
            #print("Unique: " + str(df[i].unique()))
            print("In " + i)
            for unique_value in range(len(df[i].unique())):
                print("Changed " + df[i].unique()[unique_value] + " to " + str(unique_value + 1))
                df[i]=df[i].replace(df[i].unique()[unique_value], unique_value + 1)

    print(df)


if __name__ == "__main__":
    main()
