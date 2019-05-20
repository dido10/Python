#17.ders


import numpy as np
import pandas as pd
import matplotlib.pyplot as mp

df=pd.read_csv("polynomial-regression.csv",sep=";")
#values seriyi arraye dönüştürür.
#reshape serinin x(15,) görünümünü (15,1) haline getirir.
#sklearn lib de kullanıolabilmesi için gereklidir.
y=df.araba_max_hiz.values.reshape(-1,1)
x=df.araba_fiyat.values.reshape(-1,1)

mp.scatter(x,y)
mp.xlabel("araç hızı")
mp.ylabel("araç fiyatı")
mp.show()

#linear reg.
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x,y)

#predict
y_head=lr.predict(x)
mp.plot(x,y_head,color="red")
#polynomial reg. y=bo+b1*x+b2*x^2+b3*x^3....+bn*x^n

#%%
from sklearn.preprocessing import PolynomialFeatures
#polynomial Regression degree=n
pol_reg=PolynomialFeatures(degree=2)
#fit sadece veriyi kullanır fakat fit_transform veri kullan ve 2. derece polinoma a çevir
x_poly=pol_reg.fit_transform(x)
linear_reg2=LinearRegression()
linear_reg2.fit(x_poly,y)


y_head2=linear_reg2.predict(x_poly)
mp.plot(x,y_head2,color="green",label="polinom")
mp.legend()
mp.show()

#%% 
#♦ degree değiştirerek noktalara daha da yakınsayan bir grafik oluşuyor.
pol_reg=PolynomialFeatures(degree=4)
#fit sadece veriyi kullanır fakat fit_transform veri kullan ve 2. derece polinoma a çevir
x_poly=pol_reg.fit_transform(x)
linear_reg3=LinearRegression()
linear_reg3.fit(x_poly,y)


y_head3=linear_reg3.predict(x_poly)
mp.plot(x,y_head3,color="green",label="polinom")
mp.legend()
mp.show()













