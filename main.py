import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) ucitavanje
data = pd.read_csv('dataset.csv')

# 2) prikaz prvih i poslednjih 5 + statistike atributa
print('prvih 5')
print(data.head())
print('poslednjih 5')
print(data.tail())
print('INFO:')
print(data.info())
print('DESKRIPCIJE:')
print(data.describe().T)


# 3) nedostajuce vrijednosti

nanovi = data.isna()

print('broj nedostajucih elemenata po atributu:')
print(nanovi.sum())

data.Teacher_Quality = data.Teacher_Quality.fillna(data.Teacher_Quality.mode()[0])
data.Parental_Education_Level = data.Parental_Education_Level.fillna(data.Parental_Education_Level.mode()[0])
data.Distance_from_Home = data.Distance_from_Home.fillna(data.Distance_from_Home.mode()[0])


# 4) grafici
ulaz = data.drop(columns=['Exam_Score'])
izlaz = data['Exam_Score']
numericki = ulaz.select_dtypes(include=[np.number]).columns.tolist()
kategoricki = ulaz.select_dtypes(include=["object"]).columns.tolist()

print('numericki = ', numericki)
print('kategoricki = ', kategoricki)


for i in numericki:
    plt.figure()
    plt.scatter(data[i], data['Exam_Score'])
    plt.xlabel(i)
    plt.ylabel('Exam_Score')
    plt.title('zavisnost izlaza od numerickog atributa')
    plt.show()

for kat in kategoricki:
    plt.figure()
    data.boxplot(column='Exam_Score', by=kat)
    plt.title('zavisnost izlaza od kategorickog atributa')
    plt.xlabel(kat)
    plt.ylabel('Exam_Score')
    plt.show()

# 5) endokovanje kategorickih u OH

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, dtype=int)
katOH = ohe.fit_transform(ulaz[kategoricki])
katOHimena = ohe.get_feature_names_out(kategoricki)
ulaz = ulaz.drop(columns=kategoricki)
ulaz = ulaz.join(pd.DataFrame(katOH, columns=katOHimena, index=ulaz.index))


# 6) normalizacija numerickih parametara

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
ulaz[numericki] = scaler.fit_transform(ulaz[numericki])


# 7) biranje korisnih atributa

d = ulaz.copy()
d["Exam_Score"] = izlaz
corr = d.corr(numeric_only=True)["Exam_Score"].abs()
print(corr)

izbaceni = corr[corr < 0.03].index.tolist()

print('izbaceni:')
print(izbaceni)

ulaz = ulaz.drop(columns=izbaceni)

# 8) train test split

from sklearn.model_selection import train_test_split

ulazTren, ulazTest, izlazTren, izlazTest = train_test_split(ulaz, izlaz, train_size=0.8, random_state=42)

# 9) KNN

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=7)
knn.fit(ulazTren, izlazTren)

predKNN = knn.predict(ulazTest)


# 10) lin regresija

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(ulazTren, izlazTren)

predLR = lr.predict(ulazTest)


# 11) prikaz parametara modela i MAE/MSE greske

print('hiperparametri KNN:')
print(knn.get_params())
print('hiperparametri LR')
print(lr.get_params())
print('nauceni parametri LR:')
print("intercept:", lr.intercept_)
print("coef:", lr.coef_)

from sklearn.metrics import mean_squared_error, mean_absolute_error

maeKNN = mean_absolute_error(izlazTest, predKNN)
mseKNN = mean_squared_error(izlazTest, predKNN)
mseLR = mean_squared_error(izlazTest, predLR)
maeLR = mean_absolute_error(izlazTest, predLR)

print('mse KNN: ', mseKNN)
print('mae KNN: ', maeKNN)
print('mse LR: ', mseLR)
print('mae LR: ', maeLR)


# 12) poredjenje

'''
lin regresija daje odlicne rezultate, KNN daje dobre rezultate 
'''

