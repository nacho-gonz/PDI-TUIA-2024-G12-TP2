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

B = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
Aclau = cv2.morphologyEx(bin_s, cv2.MORPH_CLOSE, B)

B = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
Aop = cv2.morphologyEx(Aclau, cv2.MORPH_OPEN, B)


kernel = np.ones((3, 3), np.uint8)
apertura = cv2.morphologyEx(Aop, cv2.MORPH_OPEN, kernel)

contours, hierarchy = cv2.findContours(Aop, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

contornos_circulo_areas  = []
contornos_solos = []


for contour in contours:
    a = cv2.contourArea(contour)
    p_c = cv2.arcLength(contour, True)**2
    fdf = a/p_c
    if fdf > 0.062:
        contornos_circulo_areas .append((contour, a))
        contornos_solos.append(contour)

moneditas = cv2.cvtColor(monedas_copy, cv2.COLOR_BGR2RGB)

contornos_ordenada = sorted(contornos_circulo_areas , key=lambda x: x[1])

max_area = contornos_ordenada[-1][1]
min_area = contornos_ordenada[0][1]

if min_area / max_area < 0.85:
    monedas_10_50 = 0
    monedas_10_peso = 0
    monedas_50 = 0
    monedas_peso = 0

    rel_50_10_max , rel_50_10_min = 0.61, 0.4
    rel_50_1_max , rel_50_1_min = 0.83, 0.74
    rel_1_10_max , rel_1_10_min = 0.73, 0.62

    for moneda, area in contornos_circulo_areas:
        if rel_50_10_min <= area/max_area <= rel_50_10_max:
            monedas_10_50 += 1
        elif rel_50_1_min <= area/max_area <= rel_50_1_max:
            monedas_peso += 1
        elif rel_1_10_min <= area/max_area <= rel_1_10_max:
            monedas_10_peso += 1
        elif area/max_area >= 0.84:
            if monedas_10_50 >= 1:
                monedas_50 += 1
            elif monedas_10_peso >= 1:
                monedas_peso +=1
            elif monedas_peso >= 1:
                monedas_50 += 1
    if monedas_10_peso == 0:
        monedas_10_peso = monedas_10_50
    elif monedas_10_50 == 0:
        monedas_10_50 = monedas_10_peso

else:
    monedas_10_peso = 0
    monedas_50 = 0
    monedas_peso = 0
    rel_recuadro_50_max , rel_recuadro_50_min = 0.97, 0.65

    for contorno in contornos_solos:
        x, y, w, h = cv2.boundingRect(contorno)

        imagen_recortada = moneditas[y-10:y+h, x:x+w]

        plt.imshow(imagen_recortada), plt.show()

        moneditas_cielab = cv2.cvtColor(imagen_recortada, cv2.COLOR_BGR2LAB)

        lab_monedita, a_monedita, b_monedita = cv2.split(moneditas_cielab)

        retval, bin_h_monedita = cv2.threshold(b_monedita, 115, 255, cv2.THRESH_BINARY)
        bin_h_monedita_not = np.bitwise_not(bin_h_monedita)

        B_monedita = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
        Aclau_monedita = cv2.morphologyEx(bin_h_monedita_not, cv2.MORPH_CLOSE, B_monedita)

        B_monedita = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        Aop_monedita = cv2.morphologyEx(Aclau_monedita, cv2.MORPH_OPEN, B_monedita)

        B_monedita = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        clau_final = cv2.morphologyEx(Aop_monedita, cv2.MORPH_CLOSE, B_monedita)

        contours_monedita, hierarchy = cv2.findContours(clau_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours_monedita) == 0:
            monedas_10_peso += 1
        else:
            area_moneda = cv2.contourArea(contours_monedita[0])
            area_recuadro = w*h
            if rel_recuadro_50_min < area_moneda/area_recuadro < rel_recuadro_50_max:
                monedas_50 += 1
            else:
                monedas_peso += 1

print(monedas_50, monedas_peso, monedas_10_peso)
