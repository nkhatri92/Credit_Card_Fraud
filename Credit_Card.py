
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
from sklearn.ensemble import RandomForestClassifier
#%matplotlib inline

df = pd.read_csv("d://Python Programs//creditcard.csv")
print(df.shape)

from sklearn.preprocessing import StandardScaler
# df["Normalized Amount"] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
# df.drop(['Amount'], axis=1, inplace=True)
# df.head()


sns.countplot("Class",data=df)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

def train_test_data(df):

    features = df.ix[:, df.columns !='Class']
    predictor = df.ix[:, df.columns == 'Class']

    train_features, test_features, train_predictor, test_predictor = train_test_split(features, predictor, test_size= 0.3, random_state=0)
    print(train_features.shape, test_features.shape, train_predictor.shape, test_predictor.shape)

    #print(train_features, test_features, train_predictor, test_predictor)
    return(train_features, test_features, train_predictor, test_predictor)

train_features, test_features, train_predictor, test_predictor = train_test_data(df)
# print(train_predictor)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=0)

smote_features_train, smote_predictor_train = smote.fit_sample(train_features,train_predictor)
smote_features_train = pd.DataFrame(data=smote_features_train, columns=train_features.columns)
smote_predictor_train = pd.DataFrame(data=smote_predictor_train,columns=["Class"])

print("Proportion of Normal data in oversampled data is ",len(smote_predictor_train[smote_predictor_train["Class"]==0])/len(smote_features_train))
print("Proportion of fraud data in oversampled data is ",len(smote_predictor_train[smote_predictor_train["Class"]==1])/len(smote_features_train))

smote_features_train["Normalized Amount"] = StandardScaler().fit_transform(smote_features_train['Amount'].values.reshape(-1, 1))
smote_features_train.drop(["Amount"],axis=1,inplace=True)
test_features["Normalized Amount"] = StandardScaler().fit_transform(test_features['Amount'].values.reshape(-1, 1))
test_features.drop(["Amount"],axis=1,inplace=True)

def analytics(model, train_features, test_features, train_predictor, test_predictor):
    clf = model
    model.fit(train_features, train_predictor.values.ravel())
    pred = clf.predict(test_features)
    cnf_matrix=confusion_matrix(test_predictor,pred)
    print(cnf_matrix)
    print("the recall for this model is :",cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))
    fig= plt.figure(figsize=(6,3))# to plot the graph
    print("TP",cnf_matrix[1,1,]) # no of fraud transaction which are predicted fraud
    print("TN",cnf_matrix[0,0]) # no. of normal transaction which are predited normal
    print("FP",cnf_matrix[0,1]) # no of normal transaction which are predicted fraud
    print("FN",cnf_matrix[1,0]) # no of fraud Transaction which are predicted normal
    sns.heatmap(cnf_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    print("\n----------Classification Report------------------------------------")
    print(classification_report(test_predictor,pred))



#clf = LogisticRegression()
clf = RandomForestClassifier()
#analytics(clf, train_features, test_features, train_predictor, test_predictor )

analytics(clf, smote_features_train, test_features, smote_predictor_train, test_predictor)
