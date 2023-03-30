from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd 

def main():
    print("Hello world")
    df = load_data()
    df = prepare_data(df)
    classification(df)

def load_data():
    dataset = "exams.csv"
    path = "data/" + dataset

    dataset_names = ["gender","ethnicity","parental_education","lunch","test_prep_course","math_score","reading_score","writing_score"]

    df = pd.read_csv(path, names=dataset_names) 
    return df

def prepare_data(df):
    df.drop(index=df.index[0], axis=0, inplace=True)
    #print(df)
    for i in df.columns:     
        #print(i)
        try:
            df[i] = pd.to_numeric(df[i])
        except Exception as e:
            print("In " + i)
            for unique_value in range(len(df[i].unique())):
                print("Changed " + df[i].unique()[unique_value] + " to " + str(unique_value + 1))
                df[i]=df[i].replace(df[i].unique()[unique_value], unique_value + 1)

    return df

def classification(df):
    #KNN
    X = df.iloc[:, 1:].values
    Y = df.iloc[:, 0].values 

    #print(X)
    #print(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25) 

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test) 

    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, Y_train) 

    Y_predict = classifier.predict(X_test)

    print(confusion_matrix(Y_test, Y_predict))
    print(classification_report(Y_test, Y_predict)) 
    print(accuracy_score(Y_test, Y_predict))

if __name__ == "__main__":
    main()
