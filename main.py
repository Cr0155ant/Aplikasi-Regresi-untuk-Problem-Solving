import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# Pembaca File Yang Diunggah
file_path = r'C:\Users\User\Documents\Tugas\Student_Performance.csv'
data = pd.read_csv(file_path)

# Menampilkan Nama Kolom Yang Sesuai
print(data.columns)
print(data.head())

# Memilih Kolom Yang Relevan Dengan Nama Yang Sesuai
TB = data['Hours Studied'].values
NT = data['Performance Index'].values

# Model Linear Untuk Problem 2
X_TB = TB.reshape(-1, 1)
linear_model_TB = LinearRegression()
linear_model_TB.fit(X_TB, NT)
NT_pred_linear_TB = linear_model_TB.predict(X_TB)

# Hasil Dengan Regresi Linear
plt.figure(figsize=(12, 6))
plt.scatter(TB, NT, color='red', label='Data Asli')
plt.plot(TB, NT_pred_linear_TB, color='blue', label='Regresi Linear')
plt.title('Regresi Linear (Durasi Waktu Belajar vs Nilai Ujian)')
plt.xlabel('Durasi Waktu Belajar (TB)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.grid(True)
plt.show()

# Menghitung RMS Model Linear (TB)
rms_linear_TB = np.sqrt(mean_squared_error(NT, NT_pred_linear_TB))
print(f'RMS untuk model linear (TB): {rms_linear_TB}')

# Model Eksponensial Untuk Problem 2
def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

popt, _ = curve_fit(exp_func, TB, NT, maxfev = 1000000)
NT_pred_exp_TB = exp_func(TB, *popt)

# Hasil Dengan Regresi Eksponensial
plt.figure(figsize=(12, 6))
plt.scatter(TB, NT, color='red', label='Data Asli')
plt.plot(TB, NT_pred_exp_TB, color='blue', label='Regresi Eksponensial')
plt.title('Regresi Eksponensial (Durasi Waktu Belajar vs Nilai Ujian)')
plt.xlabel('Jumlah Latihan Soal (TB)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.grid(True)
plt.show()

# Menghitung RMS Model Eksponensial (TB)
rms_exp_TB = np.sqrt(mean_squared_error(NT, NT_pred_exp_TB))
print(f'RMS untuk model eksponensial (TB): {rms_exp_TB}')
