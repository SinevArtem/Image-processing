import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, mean_squared_error
import copy

image = cv2.imread('img.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Гауссов шум
mean = 0
stddev = 100
noise_gauss = np.zeros(image_gray.shape, np.uint8)
cv2.randn(noise_gauss, mean, stddev)
image_noise_gauss = cv2.add(image_gray, noise_gauss)
plt.imshow(image_noise_gauss, cmap="gray")
plt.show()

# 2. Постоянный шум
noise = np.random.randint(0, 101, size=(image_gray.shape[0], image_gray.shape[1]), dtype=int)
zeros_pixel = np.where(noise == 0)
ones_pixel = np.where(noise == 100)

image_sp = copy.deepcopy(image_gray)
image_sp[zeros_pixel] = 0
image_sp[ones_pixel] = 255

plt.imshow(image_sp, cmap="gray")
plt.show()

# Оценка зашумленных изображений
print("=== Оценка зашумленных изображений ===")
mse_gauss = mean_squared_error(image_gray, image_noise_gauss)
ssim_gauss = structural_similarity(image_gray, image_noise_gauss)
print(f"Гауссов шум: MSE = {mse_gauss:.2f}, SSIM = {ssim_gauss:.4f}")

mse_sp = mean_squared_error(image_gray, image_sp)
ssim_sp = structural_similarity(image_gray, image_sp)
print(f"Шум соль-перец: MSE = {mse_sp:.2f}, SSIM = {ssim_sp:.4f}")

# Тестирование фильтров для гауссова шума
print("\n=== ФИЛЬТРАЦИЯ ГАУССОВА ШУМА ===")

# Медианный фильтр с разными параметрами
image_gauss_median3 = cv2.medianBlur(image_noise_gauss, 3)
image_gauss_median5 = cv2.medianBlur(image_noise_gauss, 5)
image_gauss_median7 = cv2.medianBlur(image_noise_gauss, 7)

# Гауссов фильтр с разными параметрами
image_gauss_gauss3 = cv2.GaussianBlur(image_noise_gauss, (3, 3), 0)
image_gauss_gauss5 = cv2.GaussianBlur(image_noise_gauss, (5, 5), 0)
image_gauss_gauss7 = cv2.GaussianBlur(image_noise_gauss, (7, 7), 0)

# Билатеральный фильтр с разными параметрами
image_gauss_bilat1 = cv2.bilateralFilter(image_noise_gauss, 5, 50, 50)
image_gauss_bilat2 = cv2.bilateralFilter(image_noise_gauss, 9, 75, 75)
image_gauss_bilat3 = cv2.bilateralFilter(image_noise_gauss, 15, 100, 100)

# Фильтр нелокальных средних с разными параметрами
image_gauss_nlm1 = cv2.fastNlMeansDenoising(image_noise_gauss, h=10)
image_gauss_nlm2 = cv2.fastNlMeansDenoising(image_noise_gauss, h=20)
image_gauss_nlm3 = cv2.fastNlMeansDenoising(image_noise_gauss, h=30)

# Оценка результатов для гауссова шума
filters_gauss = {
    'Медианный 3x3': image_gauss_median3,
    'Медианный 5x5': image_gauss_median5,
    'Медианный 7x7': image_gauss_median7,
    'Гауссов 3x3': image_gauss_gauss3,
    'Гауссов 5x5': image_gauss_gauss5,
    'Гауссов 7x7': image_gauss_gauss7,
    'Билатеральный (5,50,50)': image_gauss_bilat1,
    'Билатеральный (9,75,75)': image_gauss_bilat2,
    'Билатеральный (15,100,100)': image_gauss_bilat3,
    'NLM h=10': image_gauss_nlm1,
    'NLM h=20': image_gauss_nlm2,
    'NLM h=30': image_gauss_nlm3
}

best_ssim_gauss = 0
best_filter_gauss = ""

for name, filtered in filters_gauss.items():
    mse = mean_squared_error(image_gray, filtered)
    ssim_val = structural_similarity(image_gray, filtered)
    print(f"{name:25}: MSE = {mse:6.2f}, SSIM = {ssim_val:.4f}")

    if ssim_val > best_ssim_gauss:
        best_ssim_gauss = ssim_val
        best_filter_gauss = name

print(f"\nЛучший фильтр для гауссова шума: {best_filter_gauss} (SSIM = {best_ssim_gauss:.4f})")

# Тестирование фильтров для шума соль-перец
print("\n=== ФИЛЬТРАЦИЯ ШУМА СОЛЬ-ПЕРЕЦ ===")

# Медианный фильтр
image_sp_median3 = cv2.medianBlur(image_sp, 3)
image_sp_median5 = cv2.medianBlur(image_sp, 5)
image_sp_median7 = cv2.medianBlur(image_sp, 7)

# Гауссов фильтр
image_sp_gauss3 = cv2.GaussianBlur(image_sp, (3, 3), 0)
image_sp_gauss5 = cv2.GaussianBlur(image_sp, (5, 5), 0)
image_sp_gauss7 = cv2.GaussianBlur(image_sp, (7, 7), 0)

# Билатеральный фильтр
image_sp_bilat1 = cv2.bilateralFilter(image_sp, 5, 50, 50)
image_sp_bilat2 = cv2.bilateralFilter(image_sp, 9, 75, 75)
image_sp_bilat3 = cv2.bilateralFilter(image_sp, 15, 100, 100)

# Фильтр нелокальных средних
image_sp_nlm1 = cv2.fastNlMeansDenoising(image_sp, h=10)
image_sp_nlm2 = cv2.fastNlMeansDenoising(image_sp, h=20)
image_sp_nlm3 = cv2.fastNlMeansDenoising(image_sp, h=30)

# Оценка результатов для шума соль-перец
filters_sp = {
    'Медианный 3x3': image_sp_median3,
    'Медианный 5x5': image_sp_median5,
    'Медианный 7x7': image_sp_median7,
    'Гауссов 3x3': image_sp_gauss3,
    'Гауссов 5x5': image_sp_gauss5,
    'Гауссов 7x7': image_sp_gauss7,
    'Билатеральный (5,50,50)': image_sp_bilat1,
    'Билатеральный (9,75,75)': image_sp_bilat2,
    'Билатеральный (15,100,100)': image_sp_bilat3,
    'NLM h=10': image_sp_nlm1,
    'NLM h=20': image_sp_nlm2,
    'NLM h=30': image_sp_nlm3
}

best_ssim_sp = 0
best_filter_sp = ""

for name, filtered in filters_sp.items():
    mse = mean_squared_error(image_gray, filtered)
    ssim_val = structural_similarity(image_gray, filtered)
    print(f"{name:25}: MSE = {mse:6.2f}, SSIM = {ssim_val:.4f}")

    if ssim_val > best_ssim_sp:
        best_ssim_sp = ssim_val
        best_filter_sp = name

print(f"\nЛучший фильтр для шума соль-перец: {best_filter_sp} (SSIM = {best_ssim_sp:.4f})")

# Визуализация лучших результатов
plt.figure(figsize=(15, 10))

# Исходное изображение
plt.subplot(2, 4, 1)
plt.imshow(image_gray, cmap="gray")
plt.title("Исходное изображение")
plt.axis('off')

# Зашумленные изображения
plt.subplot(2, 4, 2)
plt.imshow(image_noise_gauss, cmap="gray")
plt.title("Гауссов шум")
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(image_sp, cmap="gray")
plt.title("Шум соль-перец")
plt.axis('off')

# Лучшие результаты для гауссова шума
plt.subplot(2, 4, 5)
best_gauss_img = filters_gauss[best_filter_gauss]
plt.imshow(best_gauss_img, cmap="gray")
plt.title(f"Лучший для гауссова: {best_filter_gauss}")
plt.axis('off')

# Лучшие результаты для шума соль-перец
plt.subplot(2, 4, 6)
best_sp_img = filters_sp[best_filter_sp]
plt.imshow(best_sp_img, cmap="gray")
plt.title(f"Лучший для соль-перец: {best_filter_sp}")
plt.axis('off')

# Сравнение медианного фильтра для обоих типов шума
plt.subplot(2, 4, 7)
plt.imshow(image_gauss_median5, cmap="gray")
plt.title("Медианный 5x5 (гауссов)")
plt.axis('off')

plt.subplot(2, 4, 8)
plt.imshow(image_sp_median5, cmap="gray")
plt.title("Медианный 5x5 (соль-перец)")
plt.axis('off')

plt.tight_layout()
plt.show()
