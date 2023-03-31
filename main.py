from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd 
import matplotlib.pyplot as plt
import sys

def main():
    try:
        function_to_use = sys.argv[1]
    except:
        print("Add parameter 'find_best_k' or 'classification' to run this script")
        return
    
    if(function_to_use == "find_best_K"):
        df = load_data()
        df = prepare_data(df)
        find_best_K(df)
    elif(function_to_use == "classification"):
        try:
            K = sys.argv[2]
        except:
            print("Add additional parameter, K-neighbors number, must be integer")
            return
        try:
            K = int(K)
        except:
            print(K + " is not integer")
            return
        df = load_data()
        df = prepare_data(df)
        classification(df, K)
    else:
        print("Type correct parameter")

def load_data():
    dataset = "exams.csv"
    path = "data/" + dataset

    #dataset_names = ["gender","ethnicity","parental_education","lunch","test_prep_course","math_score","reading_score","writing_score"]

    #df = pd.read_csv(path, names=dataset_names) 
    df = pd.read_csv(path) 
    return df

def prepare_data(df):
    df.drop(index=df.index[0], axis=0, inplace=True)
    print("Data preparation\n----------")
    #print(df)
    for i in df.columns:
        #print(i)
        try:
            df[i] = pd.to_numeric(df[i])
        except Exception as e:
            print("In " + i)
            for unique_value in range(len(df[i].unique())):
                print("\tChanged " + df[i].unique()[unique_value] + " to " + str(unique_value + 1))
                df[i]=df[i].replace(df[i].unique()[unique_value], unique_value + 1)
    print("----------\n")
    return df

def find_best_K(df):
    #KNN
    print("Testing K in range 1 to 25")
    X = df.iloc[:, 1:].values
    Y = df.iloc[:, 0].values 

    #print(X)
    #print(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25) 

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test) 

    classifier_acc = []
    max_k = 25
    repeat_each_k = 100

    for i in range(1, max_k + 1):
        print("K = " + str(i))
        classifier_acc_temp = []
        for j in range(1, repeat_each_k + 1):
            classifier = KNeighborsClassifier(n_neighbors=i)
            classifier.fit(X_train, Y_train) 

            Y_predict = classifier.predict(X_test)

            classifier_acc_temp.append(accuracy_score(Y_test, Y_predict))
        classifier_acc.append(sum(classifier_acc_temp) / len(classifier_acc_temp))

    #print(confusion_matrix(Y_test, Y_predict))
    #print(classification_report(Y_test, Y_predict)) 
    #print(accuracy_score(Y_test, Y_predict))

    plt.figure(figsize=(10,6))
    plt.plot(range(1,max_k + 1),classifier_acc,color='black', marker='o',markerfacecolor='red', markersize=10)
    plt.title('Accuracy per K-neighbors')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()

def classification(df, K):
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

    classifier = KNeighborsClassifier(n_neighbors=K)
    classifier.fit(X_train, Y_train) 

    Y_predict = classifier.predict(X_test)

    #print(confusion_matrix(Y_test, Y_predict))
    #print(classification_report(Y_test, Y_predict)) 
    acc_score = accuracy_score(Y_test, Y_predict)
    print("Accuracy: " + str(acc_score))

if __name__ == "__main__":
    main()
