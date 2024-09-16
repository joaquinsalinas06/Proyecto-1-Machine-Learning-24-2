import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


def RMSE(Y, Y_pred):
    return np.sqrt(mean_squared_error(Y,Y_pred))

#def MAD(Y, Y_pred):
    #return np.mean(abs(Y-Y_pred))

#def adjusted_r2(r2,n,p):
    #return 1 - (1-r2)*((n-1)/(n-p-1))


df = pd.read_csv('./forestfires.csv')

#print(df.head)

df = pd.get_dummies(df, columns=['month', 'day'], drop_first=True)

#print(df.head)
df['log_area'] = np.log(df['area'] + 1)

#sns.histplot(df['log_area'])
#plt.show()

X = df.iloc[:, :-1]
y = df['log_area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


rmse_scorer = make_scorer(RMSE, greater_is_better= False)
mad_scorer = make_scorer(lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred)))

#---------------------------- REGRESION LINEAL MULTIPLE

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_y_pred = lasso_model.predict(X_test)

n = len(X_train)
p = X_train.shape[1]

#Cross validation

mad_scores_lineal = cross_val_score(lasso_model, X_train, y_train, cv=10, scoring=mad_scorer)
rmse_scores_lineal = cross_val_score(lasso_model, X_train, y_train, cv=10, scoring=rmse_scorer)

print(f"MAD por fold: {mad_scores_lineal}")
print(f"Promedio MAD: {np.mean(mad_scores_lineal)}")
print(f"Desviación estándar MAD: {np.std(mad_scores_lineal)}")

print(f"RMSE por fold: {rmse_scores_lineal}")
print(f"Promedio RMSE: {np.mean(rmse_scores_lineal)}")
print(f"Desviación estándar RMSE: {np.std(rmse_scores_lineal)}")


#-------------------------- CARACTERISTICAS MAS RELEVANTES CON LASSO

#coef = pd.Series(lasso_model.coef_, index=X_train.columns)
#print(coef)

#relevant_features = coef[coef != 0]
#print("Características más relevantes:", relevant_features.index)

#sns.barplot(x=relevant_features.values, y=relevant_features.index)
#plt.title('Características más relevantes (Lasso)')
#plt.show()

#---------------------- REGRESION POLINOMIAL CON LASSO

poly_lasso_model = make_pipeline(PolynomialFeatures(degree=3, include_bias= False), StandardScaler(), Lasso(alpha=0.6,max_iter= 10000))

poly_lasso_model.fit(X_train, y_train)

poly_lasso_y_pred = poly_lasso_model.predict(X_test)

#Cross validation
mad_scores_polinomial = cross_val_score(poly_lasso_model, X_train, y_train, cv=10, scoring=mad_scorer)
rmse_scores_polinomial = cross_val_score(poly_lasso_model, X_train, y_train, cv=10, scoring=rmse_scorer)

print(f"MAD por fold: {mad_scores_polinomial}")
print(f"Promedio MAD: {np.mean(mad_scores_polinomial)}")
print(f"Desviación estándar MAD: {np.std(mad_scores_polinomial)}")

print(f"RMSE por fold: {rmse_scores_polinomial}")
print(f"Promedio RMSE: {np.mean(rmse_scores_polinomial)}")
print(f"Desviación estándar RMSE: {np.std(rmse_scores_polinomial)}")

#-------------------- SUPPORT VECTOR REGRESSOR

svr_pipeline = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=9.6, gamma='scale'))

svr_pipeline.fit(X_train, y_train)

svr_y_pred = svr_pipeline.predict(X_test)

#Cross validation
mad_scores_svr = cross_val_score(svr_pipeline, X_train, y_train, cv=10, scoring=mad_scorer)
rmse_scores_svr = cross_val_score(svr_pipeline, X_train, y_train, cv=10, scoring=rmse_scorer)

print(f"MAD por fold: {mad_scores_svr}")
print(f"Promedio MAD: {np.mean(mad_scores_svr)}")
print(f"Desviación estándar MAD: {np.std(mad_scores_svr)}")

print(f"RMSE por fold: {rmse_scores_svr}")
print(f"Promedio RMSE: {np.mean(rmse_scores_svr)}")
print(f"Desviación estándar RMSE: {np.std(rmse_scores_svr)}")


plt.figure(figsize=(10, 8))

sns.scatterplot(x=y_test, y=lasso_y_pred, label='Regresión Lineal Múltiple')

sns.scatterplot(x=y_test, y=poly_lasso_y_pred, label='Regresión Polinomial')

sns.scatterplot(x=y_test, y=svr_y_pred, label='Support vector Regression')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Línea ideal (y=x)')

plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Comparación de Predicciones vs Valores Reales')
plt.legend()
#plt.show()

