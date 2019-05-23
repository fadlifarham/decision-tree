#%%
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

def decision_tree(data, label):
    le_red = preprocessing.LabelEncoder()
    le_green = preprocessing.LabelEncoder()
    le_blue = preprocessing.LabelEncoder()
    le_label = preprocessing.LabelEncoder()

    # print(data['RED'].values)

    data['RED'] = le_red.fit_transform(data['RED'])
    data['GREEN'] = le_green.fit_transform(data['GREEN'])
    data['BLUE'] = le_blue.fit_transform(data['BLUE'])
    label['CONTENT'] = le_label.fit_transform(label['CONTENT'])

    # X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=1)
    n = data.shape[0]
    error = 0

    # LOO
    for i in range(0, n):
        clf = tree.DecisionTreeClassifier()

        train_data = data.drop(data.index[i])
        train_label = label.drop(data.index[i])

        test_data = data.iloc[[i]]
        test_label = label.iloc[[i]]

        clf = clf.fit(train_data, train_label)
        predict = clf.predict(test_data)

        if predict[0] != test_label['CONTENT'].values[0]:
            error += 1

    print(str(error) + " error dari " + str(n) + " data")
    print("error rate : " + str(error / n * 100) + "%")

    # print(data.columns.values)
    feature_cols = list(data.columns.values)

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = feature_cols, class_names=['0','1', '2'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    graph.write_png('diabetes.png')
    Image(graph.create_png())


def main():
    pd.options.mode.chained_assignment = None
    dataset = pd.read_csv('data/data.csv')
    data = dataset.loc[:, :'BLUE']
    label = dataset.loc[:, 'CONTENT':]

    decision_tree(data, label)

if __name__ == "__main__":
    main()