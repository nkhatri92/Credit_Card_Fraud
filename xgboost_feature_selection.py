from xgboost import XGBClassifier
import pandas as pd
import matplotlib.pyplot
model = XGBClassifier()


df = pd.read_csv("d://Python Programs//creditcard.csv")

features = df.ix[:, df.columns !='Class']
predictor = df.ix[:, df.columns == 'Class']

model.fit(features, predictor)


print(model.feature_importance_)

pyplot.bar(range(len(model.feature_importance_)), model.feature_importance_)
pyplot.show()
