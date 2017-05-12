#program begins: final project

import matplotlib.pyplot as plt #reguired for plotting
import pandas as pd #required to import data and construct a data frame
import seaborn as sns#visualization library
from sklearn.model_selection import train_test_split #required to split the data set
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
from sklearn.ensemble import RandomForestClassifier #Algorithm
from sklearn.tree import DecisionTreeClassifier #Algorithm
from sklearn.linear_model import LogisticRegression #Algorithm
from sklearn.naive_bayes import GaussianNB #Algorithm
from sklearn.neighbors import KNeighborsClassifier #Algorithm
from sklearn.ensemble import GradientBoostingClassifier #Algorithm
from sklearn.preprocessing import StandardScaler# required for feature scaling
from imblearn.over_sampling import SMOTE #required for oversampling

df = pd.read_csv("d://Python Programs//creditcard.csv")#read the file in a data frame

print(df.head())#to view some data at a glance
print(df.shape)# to get the number of rows and columns

sns.countplot("Class",data=df) #to show the imabalance in the data set

def train_test_data(df): #this function splits the data set into testing and training data

    features = df.ix[:, df.columns !='Class'] #drop the class to be predicted from features
    predictor = df.ix[:, df.columns == 'Class'] #class to be predicted

    train_features, test_features, train_predictor, test_predictor = train_test_split(features, predictor, test_size= 0.3, random_state=0)

    return(train_features, test_features, train_predictor, test_predictor)

train_features, test_features, train_predictor, test_predictor = train_test_data(df) #call functin to split in training and testing

def oversampling(train_features,train_predictor): #function to implement oversampling
    smote = SMOTE(random_state=0) #object creation

    smote_features_train, smote_predictor_train = smote.fit_sample(train_features,train_predictor.values.ravel())
    smote_features_train = pd.DataFrame(data=smote_features_train, columns=train_features.columns)
    smote_predictor_train = pd.DataFrame(data=smote_predictor_train,columns=["Class"])

    print("Proportion of Normal data in oversampled data is ",len(smote_predictor_train[smote_predictor_train["Class"]==0])/len(smote_features_train))
    print("Proportion of fraud data in oversampled data is ",len(smote_predictor_train[smote_predictor_train["Class"]==1])/len(smote_features_train))

    return(smote_features_train, smote_predictor_train)
smote_features_train, smote_predictor_train = oversampling(train_features, train_predictor) #call function to implement oversampling

def normalization(smote_features_train, test_features): #function to implement normalization
    smote_features_train["Normalized Amount"] = StandardScaler().fit_transform(smote_features_train['Amount'].values.reshape(-1, 1)) #add column of normalized amount
    smote_features_train.drop(["Amount"],axis=1,inplace=True) #drop amount now

    test_features["Normalized Amount"] = StandardScaler().fit_transform(test_features['Amount'].values.reshape(-1, 1))
    test_features.drop(["Amount"],axis=1,inplace=True)

    return(smote_features_train, test_features)

def feature_drop(smote_features_train, test_features): #function to implement attribute selection
    smote_features_train.drop(["V18"],axis=1,inplace=True) #drop attribute V18 from train
    smote_features_train.drop(["V24"],axis=1,inplace=True)#drop attribute V24 from train
    test_features.drop(["V18"],axis=1,inplace=True)#drop attribute V18 from test
    test_features.drop(["V24"],axis=1,inplace=True)#drop attribute V24 from test
    return(smote_features_train, test_features)

def analytics(model, train_features, test_features, train_predictor, test_predictor):# function where the model is fit
    clf = model#object creation
    model.fit(train_features, train_predictor.values.ravel()) #fit model
    pred = clf.predict(test_features) #result
    cnf_matrix=confusion_matrix(test_predictor,pred)
    recalll = cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]) #calculating recall
    fig= plt.figure(figsize=(6,3))# to plot the graph
    sns.heatmap(cnf_matrix,annot=True,linewidths=0.5) #heatmap of confusion matrix
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    print("\n----------Classification Report------------------------------------")
    print(classification_report(test_predictor,pred))

    false_positive_rate, true_positive_rate, thresholds = roc_curve(test_predictor, pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return(roc_auc, recalll)

clf1 = LogisticRegression() # object creation
clf2 = RandomForestClassifier() # object creation
clf3 = DecisionTreeClassifier(random_state=0) # object creation
clf4 = GaussianNB() # object creation
clf5 = KNeighborsClassifier() # object creation
clf6= GradientBoostingClassifier(n_estimators=200, learning_rate=1.0, max_depth=1, random_state=0) # object creation

aurocc = [] #creating empty list to store values
recall = [] #creating empty list to store values
classifiers = [clf1, clf2, clf3, clf4, clf5, clf6] #creating list of classifiers

def plot(aurocc, recall, classifiers, train_features, test_features, train_predictor, test_predictor):
    classifiers = [clf1, clf2, clf3, clf4, clf5, clf6] #they need to be initialized in the loop
    for i , clf in enumerate(classifiers): #learnt something cool here
        auroc1, recall1 = analytics(clf, train_features, test_features, train_predictor, test_predictor)
        print("The auc and recall fpr this model are", auroc1, recall1)
        aurocc.append(auroc1) #adding values to the list
        recall.append(recall1) #adding values to the list

    plt.bar(range(len(aurocc)),aurocc, tick_label = ['LR', 'RForest', 'DTree', 'NB', 'KNN', 'GBoost'])# bar plot of auc
    plt.ylabel('AUC')
    plt.show()

    plt.bar(range(len(recall)),recall, tick_label = ['LR', 'RForest', 'DTree', 'NB', 'KNN', 'GBoost'])# bar plot of recall
    plt.ylabel('Recall')
    plt.show()

j=3
while (j!=-1):# this loop iterates over the code 4 times
    if j==3:#plot when feaeture engineering not implemented
        plot(aurocc, recall, classifiers, train_features, test_features, train_predictor, test_predictor)
    elif j==2:# plotting when oversampling implemented but nomalization not implemented
        plot(aurocc, recall, classifiers, smote_features_train, test_features, smote_predictor_train, test_predictor)
    elif j==1:# plotting when oversampling and nomalization implemented
        smote_features_train, test_features = normalization(smote_features_train, test_features)
        plot(aurocc, recall, classifiers, smote_features_train, test_features, smote_predictor_train, test_predictor)
    elif j==0:# plotting when when oversampling, feature selection and nomalization implemented
        smote_features_train, test_features =  feature_drop(smote_features_train, test_features)
        plot(aurocc, recall, classifiers, smote_features_train, test_features, smote_predictor_train, test_predictor)
    aurocc = [] #making the list empty
    recall = [] #making the list empty
    j+=-1# decreasing the counter

#program ends
