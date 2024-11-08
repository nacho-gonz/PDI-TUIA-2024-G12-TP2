import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

monedas = cv2.imread('monedas.jpg')

monedas_copy = monedas.copy()

monedas_rgb = cv2.cvtColor(monedas_copy, cv2.COLOR_BGR2RGB)

monedas_blur = cv2.blur(monedas, (5,5))

img_hls = cv2.cvtColor(monedas_blur, cv2.COLOR_BGR2HLS)

h, l, s = cv2.split(img_hls)

l_smooth = cv2.blur(l, (15, 15))
s_smooth = cv2.blur(s, (15, 15))
h_smooth = cv2.blur(h, (15, 15))

img_smooth_l = cv2.cvtColor(cv2.merge((h, l_smooth, s)), cv2.COLOR_HLS2RGB)

retval, bin_s = cv2.threshold(s_smooth, 14, 255, cv2.THRESH_BINARY)

retval, bin_h = cv2.threshold(h, 60, 255, cv2.THRESH_BINARY)

plt.imshow(bin_s, cmap='gray'),plt.show()

B = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
Aclau = cv2.morphologyEx(bin_s, cv2.MORPH_CLOSE, B)

B = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
Aop = cv2.morphologyEx(Aclau, cv2.MORPH_OPEN, B)


kernel = np.ones((3, 3), np.uint8)
apertura = cv2.morphologyEx(Aop, cv2.MORPH_OPEN, kernel)

plt.imshow(apertura, cmap='gray'),plt.show()

contours, hierarchy = cv2.findContours(Aop, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

contornos_circulo = []

for contour in contours:
    a = cv2.contourArea(contour)
    p_c = cv2.arcLength(contour, True)**2
    fdf = a/p_c
    if fdf > 0.062:
        contornos_circulo.append((contour , a))



cv2.drawContours(monedas_rgb, contornos_circulo,-1, (0, 255, 0), 2)
plt.imshow(monedas_rgb),plt.show()


moneditas = cv2.cvtColor(monedas_copy, cv2.COLOR_BGR2RGB)

contornos_ordenada = sorted(contornos_circulo, key=lambda x: x[1])

max_ancho = contornos_ordenada[-1][1]
min_ancho = contornos_ordenada[0][1]

monedas_10_50 = 0
monedas_10_peso = 0
monedas_50 = 0
monedas_peso = 0

rel_50_10_max , rel_50_10_min = 0.619, 0.4
rel_50_1_max , rel_50_1_min = 0.83, 0.75
rel_1_10_max , rel_1_10_min = 0.74, 0.62

if min_ancho / max_ancho < 0.85:
    for moneda, ancho in contornos_circulo:
        if rel_50_10_min <= ancho/max_ancho <= rel_50_10_max:
            monedas_10_50 += 1
        elif rel_50_1_min <= ancho/max_ancho <= rel_50_1_max:
            monedas_peso += 1
        elif rel_1_10_min <= ancho/max_ancho <= rel_1_10_max:
            monedas_10_peso += 1
        elif ancho/max_ancho >= 0.84:
            if monedas_10_50 >= 1:
                monedas_50 += 1
            elif monedas_10_peso >= 1:
                monedas_peso +=1
            elif monedas_peso >= 1:
                monedas_50 += 1

else:
    pass