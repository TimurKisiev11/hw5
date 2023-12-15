import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import stats, ifft, fft
from scipy.signal import lfilter
from scipy.signal import savgol_filter

def delete_noise_1(data, threshold = 0.1):
    fft_data = np.fft.fft(data)
    magnitudes = np.abs(fft_data)
    max_magnitude = np.max(magnitudes)
    noise_threshold = threshold * max_magnitude
    filtered_fft = fft_data * (magnitudes > noise_threshold)
    filtered_data = np.fft.ifft(filtered_fft)
    return filtered_data.real

def delete_noise_2(y_data, threshold= 0.5):
    fourier_transform = np.fft.fft(y_data)
    freq = np.fft.fftfreq(y_data.size)
    threshold_idx = int(threshold * len(freq))
    fourier_transform[threshold_idx:-threshold_idx] = 0
    denoised_data = np.fft.ifft(fourier_transform)
    return denoised_data

def delete_noise_3(data):
    fft_result = np.fft.fft(data)
    power_spectrum = np.abs(fft_result)**2
    threshold = 0.1 * np.max(power_spectrum)
    print(threshold)
    fft_result[power_spectrum < threshold] = 0
    denoised_data = np.fft.ifft(fft_result)
    return denoised_data



def func(x, A, w, phi):
  return A*np.cos(w*x + phi)

x_data = np.linspace(0, 10, 30)
y_data = np.array([4.10465607, 1.69794745, -2.32313143, -4.07624211, -1.66543449, 2.1145771,
                   3.70880127, 2.16805654, -2.04181849, -4.07636456, -1.83702696, 1.95313412,
                   4.18439283, 1.95123149, -1.90018249, -3.88197156, -2.25644626, 1.58158325,
                   4.08242668, 2.51835565, -1.27829727, -4.0866171, -2.30645672, 1.08053092,
                   4.07574853, 2.79301888, -0.89299698, -3.91278811, -3.00365192, 0.792355])

# Применяем обратное преобразование Фурье
filtered_data = delete_noise_3(y_data)
print(y_data)
print(filtered_data.real)
parameters = curve_fit(func, x_data, filtered_data.real, p0 = (4, 3 , 0.09))
res_A, res_w, res_phi = parameters[0]
print("Амплитуда A = " + str(res_A))
print("Частота w = " + str(res_w))
print("Фаза phi = " + str(res_phi))
# Строим график искомой функции с найденными параметрами
y_fit = func(x_data, res_A, res_w, res_phi)
plt.plot(x_data, y_fit, color='red', lw=1.5, label = 'fitted')
plt.plot(x_data, y_data, c='blue', label = 'noisy data')
plt.legend()
plt.show()
# Строим график модуля разности двух графиков
y_fit = func(x_data, res_A, res_w, res_phi)
plt.plot(x_data, abs(y_fit - y_data), color='red', lw=1.5)
plt.legend()
plt.show()

