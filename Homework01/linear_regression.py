#import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

import data_processing_interface as dpi
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

rs_num = 42
# Denemek için bu değiştirilebilir

X_house = dpi.my_data["House Size (sqft)"].values
y_price = dpi.my_data["Price ($)"].values
y_log = dpi.my_data["Log Price ($)"].values

X_bedroom = dpi.my_data["Number of Bedrooms"].values
y_bathroom = dpi.my_data["Number of Bathrooms"].values

# Alttaki conditional'larımız, eğer atadığımız değerler 1 boyutlu array ise
# (ki öyleler), onları otomatikman .reshape ile 2 boyutlu array'e çevirip yeniden atıyor.

if X_house.ndim == 1:
    X_house = X_house.reshape((-1, 1))

if y_price.ndim == 1:
    y_price = y_price.reshape((-1, 1))

if y_log.ndim == 1:
    y_log = y_log.reshape((-1, 1))

if X_bedroom.ndim == 1:
    X_bedroom = X_bedroom.reshape((-1, 1))

if y_bathroom.ndim == 1:
    y_bathroom = y_bathroom.reshape((-1, 1))


# Datasetimizde, farklı sütunlarda, nümerik değeri 1 ile 3600 arasında değişebilen büyük bir aralık olduğu için
# datasetimizde Standard Scaling ile de fit_train yapmayı düşündüm. Bir Support Vector Regression örneğinde bunu yapmak
# gerekiyordu, ama scikit'in Linear Regression fonksiyonlarında önceden yapmak gerekiyor
# muydu hatırlayamadım. Böylece farkı daha iyi gözlemleyebiliriz.

scaler = StandardScaler()

X_house_scaled = scaler.fit_transform(X_house)
y_price_scaled = scaler.fit_transform(y_price)
y_log_scaled = scaler.fit_transform(y_log)
X_bedroom_scaled = scaler.fit_transform(X_bedroom)
y_bathroom_scaled = scaler.fit_transform(y_bathroom)


secim2=input("Shape'leri print'le? (Y/N)")
if secim2 == "Y":
    print("\nX_house: ")
    print(X_house.shape)
    print(type(X_house))

    print("\ny_price: ")
    print(y_price.shape)
    print(type(y_price))

    print("\ny_log: ")
    print(y_log.shape)
    print(type(y_log))

    print("\nX_bedroom: ")
    print(X_bedroom.shape)
    print(type(X_bedroom))

    print("\ny_bathroom: ")
    print(y_bathroom.shape)
    print(type(y_bathroom))


X_house_train, X_house_test, y_price_train, y_price_test = train_test_split(X_house, y_price, test_size=0.3, random_state=rs_num)

# Aynı "X_house" dependent variable'ını kullanırken "y_log"u da train-test için ayırıyoruz:
_, _, y_log_train, y_log_test = train_test_split(X_house, y_log, test_size=0.3, random_state=rs_num)

X_bedroom_train, X_bedroom_test, y_bathroom_train, y_bathroom_test = train_test_split(X_bedroom, y_bathroom, test_size=0.3, random_state=rs_num)

# Standard Scale yapılan kümelerimizi de train-test split'liyoruz:
X_house_train_scaled, X_house_test_scaled, y_price_train_scaled, y_price_test_scaled = train_test_split(X_house_scaled, y_price_scaled, test_size=0.3, random_state=rs_num)

_, _, y_log_train_scaled, y_log_test_scaled = train_test_split(X_house_scaled, y_log_scaled, test_size=0.3, random_state=rs_num)

X_bedroom_train_scaled, X_bedroom_test_scaled, y_bathroom_train_scaled, y_bathroom_test_scaled = train_test_split(X_bedroom_scaled, y_bathroom_scaled, test_size=0.3, random_state=rs_num)


lm_house_price = LinearRegression()
lm_house_log = LinearRegression()
lm_bedroom_bathroom = LinearRegression()

lm_house_price_scaled = LinearRegression()
lm_house_log_scaled = LinearRegression()
lm_bedroom_bathroom_scaled = LinearRegression()

lm_house_price.fit(X_house_train, y_price_train)
lm_house_log.fit(X_house_train, y_log_train)
lm_bedroom_bathroom.fit(X_bedroom_train, y_bathroom_train)

lm_house_price_scaled.fit(X_house_train_scaled, y_price_train_scaled)
lm_house_log_scaled.fit(X_house_train_scaled, y_log_train_scaled)
lm_bedroom_bathroom_scaled.fit(X_bedroom_train_scaled, y_bathroom_train_scaled)

y_price_pred = lm_house_price.predict(X_house_test)
y_log_pred = lm_house_log.predict(X_house_test)
y_bathroom_pred = lm_bedroom_bathroom.predict(X_bedroom_test)

y_price_pred_scaled = lm_house_price_scaled.predict(X_house_test_scaled)
y_log_pred_scaled = lm_house_log_scaled.predict(X_house_test_scaled)
y_bathroom_pred_scaled = lm_bedroom_bathroom_scaled.predict(X_bedroom_test_scaled)

print("\nUnscaled Model: House Size -> Price")
print("Standard Scale'siz ev boyutu kullanarak fiyat tahmini:")
print("Coefficients:", lm_house_price.coef_)
print("Intercept:", lm_house_price.intercept_)
print("R^2 Score:", lm_house_price.score(X_house_test, y_price_test))

print("\nUnscaled Model: House Size -> Log Price")
print("Standard Scale'siz ev boyutu kullanarak log dönüşümü yapılmış fiyat tahmini:")
print("Coefficients:", lm_house_log.coef_)
print("Intercept:", lm_house_log.intercept_)
print("R^2 Score:", lm_house_log.score(X_house_test, y_log_test))

print("\nUnscaled Model: Number of Bedrooms -> Number of Bathrooms")
print("Standard Scale'siz yatak odası sayısı kullanarak banyo sayısı tahmini:")
print("Coefficients:", lm_bedroom_bathroom.coef_)
print("Intercept:", lm_bedroom_bathroom.intercept_)
print("R^2 Score:", lm_bedroom_bathroom.score(X_bedroom_test, y_bathroom_test))

print("\nScaled Model: House Size -> Price")
print("Standard Scale'li ev boyutu kullanarak fiyat tahmini:")
print("Coefficients:", lm_house_price_scaled.coef_)
print("Intercept:", lm_house_price_scaled.intercept_)
print("R^2 Score:", lm_house_price_scaled.score(X_house_test_scaled, y_price_test_scaled))

print("\nScaled Model: House Size -> Log Price")
print("Standard Scale'li ev boyutu kullanarak log dönüşümü yapılmış fiyat tahmini:")
print("Coefficients:", lm_house_log_scaled.coef_)
print("Intercept:", lm_house_log_scaled.intercept_)
print("R^2 Score:", lm_house_log_scaled.score(X_house_test_scaled, y_log_test_scaled))

print("\nScaled Model: Number of Bedrooms -> Number of Bathrooms")
print("Standard Scale'li yatak odası sayısı kullanarak banyo sayısı tahmini:")
print("Coefficients:", lm_bedroom_bathroom_scaled.coef_)
print("Intercept:", lm_bedroom_bathroom_scaled.intercept_)
print("R^2 Score:", lm_bedroom_bathroom_scaled.score(X_bedroom_test_scaled, y_bathroom_test_scaled))


fig, axs = plt.subplots(3, 2, figsize=(14, 18))

axs[0, 0].scatter(X_house_test, y_price_test, color='blue', label='Actual')
axs[0, 0].plot(X_house_test, y_price_pred, color='red', label='Predicted')
axs[0, 0].set_title('Unscaled: House Size vs. Price')
axs[0, 0].set_xlabel('House Size (sqft)')
axs[0, 0].set_ylabel('Price ($)')
axs[0, 0].legend()

axs[1, 0].scatter(X_house_test, y_log_test, color='blue', label='Actual')
axs[1, 0].plot(X_house_test, y_log_pred, color='red', label='Predicted')
axs[1, 0].set_title('Unscaled: House Size vs. Log Price')
axs[1, 0].set_xlabel('House Size (sqft)')
axs[1, 0].set_ylabel('Log Price ($)')
axs[1, 0].legend()

axs[2, 0].scatter(X_bedroom_test, y_bathroom_test, color='blue', label='Actual')
axs[2, 0].plot(X_bedroom_test, y_bathroom_pred, color='red', label='Predicted')
axs[2, 0].set_title('Unscaled: Number of Bedrooms vs. Number of Bathrooms')
axs[2, 0].set_xlabel('Number of Bedrooms')
axs[2, 0].set_ylabel('Number of Bathrooms')
axs[2, 0].legend()

axs[0, 1].scatter(X_house_test_scaled, y_price_test_scaled, color='blue', label='Actual')
axs[0, 1].plot(X_house_test_scaled, y_price_pred_scaled, color='red', label='Predicted')
axs[0, 1].set_title('Scaled: House Size vs. Price')
axs[0, 1].set_xlabel('House Size (scaled)')
axs[0, 1].set_ylabel('Price (scaled)')
axs[0, 1].legend()

axs[1, 1].scatter(X_house_test_scaled, y_log_test_scaled, color='blue', label='Actual')
axs[1, 1].plot(X_house_test_scaled, y_log_pred_scaled, color='red', label='Predicted')
axs[1, 1].set_title('Scaled: House Size vs. Log Price')
axs[1, 1].set_xlabel('House Size (scaled)')
axs[1, 1].set_ylabel('Log Price (scaled)')
axs[1, 1].legend()

axs[2, 1].scatter(X_bedroom_test_scaled, y_bathroom_test_scaled, color='blue', label='Actual')
axs[2, 1].plot(X_bedroom_test_scaled, y_bathroom_pred_scaled, color='red', label='Predicted')
axs[2, 1].set_title('Scaled: Number of Bedrooms vs. Number of Bathrooms')
axs[2, 1].set_xlabel('Number of Bedrooms (scaled)')
axs[2, 1].set_ylabel('Number of Bathrooms (scaled)')
axs[2, 1].legend()

plt.tight_layout()
plt.show()

### Çıkarımlar:
"""
Print ile gösterilen sonuçlardan anlıyoruz ki;

Standard Scale'siz ev boyutu kullanılarak fiyat tahmini yapan modelin accuracy'si: 99.51494594367946 % (+)
Standard Scale'siz ev boyutu kullanılarak log dönüşümü yapılmış fiyat tahmini yapan modelin accuracy'si: 94.42338347224218 % (+)
Standard Scale'siz yatak odası sayısı kullanılarak banyo sayısı tahmini yapan modelin accuracy'si: 72.70141578931961 % (-)

Standard Scale'li ev boyutu kullanılarak fiyat tahmini yapan modelin accuracy'si: 99.51494594367944 % (-)
Standard Scale'li ev boyutu kullanılarak log dönüşümü yapılmış fiyat tahmini yapan modelin accuracy'si: 94.42338347224217 % (-)
Standard Scale'li yatak odası sayısı kullanılarak banyo sayısı tahmini yapan modelin accuracy'si: 72.70141578931962 % (+)

"""