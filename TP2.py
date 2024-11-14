import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image

# Carga de imágenes.

monedas = cv2.imread('monedas.jpg')

monedas_copy = monedas.copy()

monedas_rgb = cv2.cvtColor(monedas_copy, cv2.COLOR_BGR2RGB)

###############################################################
###############################################################
###############################################################

# Preprocesamiento de la imágen para la detección de monedas.

monedas_gray  = cv2.cvtColor(monedas, cv2.COLOR_BGR2GRAY)

plt.imshow(monedas_gray, cmap='gray'),plt.show()

monedas_blur = cv2.blur(monedas, (5,5))

img_hls = cv2.cvtColor(monedas_blur, cv2.COLOR_BGR2HLS)

_, _, s = cv2.split(img_hls)

plt.imshow(s, cmap='gray'),plt.show()


s_blur = cv2.blur(s, (7, 7))

retval, bin_s_blur = cv2.threshold(s_blur, 12, 255, cv2.THRESH_BINARY)
plt.imshow(bin_s_blur, cmap='gray'),plt.show()

kernel_apertura_monedas = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40,40))
clausura_s_blur = cv2.morphologyEx(bin_s_blur, cv2.MORPH_CLOSE, kernel_apertura_monedas)

plt.imshow(clausura_s_blur, cmap='gray'),plt.show()
kernel_clausura_monedas = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (140,140))
clausura_apertura_s_blur = cv2.morphologyEx(clausura_s_blur, cv2.MORPH_OPEN, kernel_clausura_monedas)

plt.imshow(clausura_apertura_s_blur, cmap='gray'),plt.show()
####################################################################
####################################################################
####################################################################

# Detección de contornos, áreas, perímetros y factores de forma de las monedas.

contours, hierarchy = cv2.findContours(clausura_apertura_s_blur, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

contornos_circulo_areas = []
contornos_solos = []
monedas_rgb = cv2.cvtColor(monedas_copy, cv2.COLOR_BGR2RGB)

for contour in contours:
    a = cv2.contourArea(contour)
    p_c = cv2.arcLength(contour, True)**2
    fdf = a/p_c
    print(1/fdf)
    if 1/fdf < 15:
        contornos_circulo_areas.append((contour , a))
        contornos_solos.append(contour)

cv2.drawContours(monedas_rgb, contornos_solos,-1, (0, 255, 0), 2)
plt.imshow(monedas_rgb),plt.show()


####################################################################
####################################################################
####################################################################

# Clasificación de monedas.

moneditas = cv2.cvtColor(monedas_copy, cv2.COLOR_BGR2RGB)

contornos_ordenada = sorted(contornos_circulo_areas, key=lambda x: x[1])

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

else:
    monedas_10 = 0
    monedas_50 = 0
    monedas_peso_sola = 0
    rel_recuadro_50_max , rel_recuadro_50_min = 0.97, 0.65

    for contorno in contornos_solos:
        x, y, w, h = cv2.boundingRect(contorno)

        imagen_recortada = moneditas[y-10:y+h, x:x+w]

        plt.imshow(imagen_recortada), plt.show()

        moneditas_cielab = cv2.cvtColor(imagen_recortada, cv2.COLOR_BGR2LAB)

        lab_monedita, a_monedita, b_monedita = cv2.split(moneditas_cielab)

        # plt.imshow(b_monedita, cmap='gray'), plt.show()

        retval, bin_h_monedita = cv2.threshold(b_monedita, 115, 255, cv2.THRESH_BINARY)
        bin_h_monedita_not = np.bitwise_not(bin_h_monedita)
        # plt.imshow(bin_h_monedita_not, cmap='gray'), plt.show()

        B_monedita = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
        Aclau_monedita = cv2.morphologyEx(bin_h_monedita_not, cv2.MORPH_CLOSE, B_monedita)

        B_monedita = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        Aop_monedita = cv2.morphologyEx(Aclau_monedita, cv2.MORPH_OPEN, B_monedita)

        B_monedita = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        clau_final = cv2.morphologyEx(Aop_monedita, cv2.MORPH_CLOSE, B_monedita)

        # plt.imshow(clau_final, cmap='gray'), plt.show()


        contours_monedita, hierarchy = cv2.findContours(clau_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours_monedita) == 0:
            monedas_10 += 1
        else:
            area_moneda = cv2.contourArea(contours_monedita[0])
            area_recuadro = w*h
            if rel_recuadro_50_min < area_moneda/area_recuadro < rel_recuadro_50_max:
                monedas_50 += 1
            else:
                monedas_peso_sola += 1

# ----------------------------------------------------------------------------

def imreconstruct(marker, mask, kernel=None):
    if kernel==None:
        kernel = np.ones((3,3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatacion
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Interseccion
        if (marker == expanded_intersection).all():                         # Finalizacion?
            break                                                           #
        marker = expanded_intersection        
    return expanded_intersection

def imclearborder(img):
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    marker = img.copy()                                         # El marcador lo defino... 
    marker[1:-1,1:-1] = 0                                       # ... seleccionando solo los bordes.
    border_elements = imreconstruct(marker=marker, mask=img)    # Calculo la reconstrucción R_{f}(f_m) --> Obtengo solo los elementos que tocan el borde.
    img_cb = cv2.subtract(img, border_elements)                 # Resto dichos elementos de la imagen original.
    return img_cb



f = monedas.copy()
esc_grises = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
f_mg = cv2.morphologyEx(esc_grises, cv2.MORPH_GRADIENT, kernel,iterations=2)

elpepe2=cv2.subtract(esc_grises, f_mg)

_, binii = cv2.threshold(elpepe2, 170,255, cv2.THRESH_BINARY)

B = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
bini_ruido = cv2.dilate(binii,B)

B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
kal = cv2.morphologyEx(bini_ruido, cv2.MORPH_CLOSE, B)

plt.imshow(kal, cmap='gray'), plt.show()

contolno, pipi = cv2.findContours(kal, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


possible_plates = []
contolno_dibujito = []
num_contorno_papa = []

for i,cnt in enumerate(contolno):


    if pipi[0][i][2] != -1 and pipi[0][i][3] == -1:
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            circulitos_dado = 0
            contolno_dibujito.append(cnt)
            puntos = approx.reshape(4, 2).astype(np.float32)

            size = 100 
            pts_dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype="float32")

            M = cv2.getPerspectiveTransform(puntos, pts_dst)

            dado_warped = cv2.warpPerspective(kal, M, (size, size))
            _, dado_warped_bin = cv2.threshold(dado_warped, 100, 255, cv2.THRESH_BINARY)
            dado_warped_bin_neg = cv2.bitwise_not(dado_warped_bin)
            dado_bin_bordes = imclearborder(dado_warped_bin_neg)

            KERNELSITO = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
            cabeza = cv2.morphologyEx(dado_bin_bordes, cv2.MORPH_OPEN, KERNELSITO)

            plt.imshow(cabeza,cmap='gray'),plt.show()

            contolnito, pipi_2 = cv2.findContours(cabeza, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
            for circulito in contolnito:
                a = cv2.contourArea(circulito)
                per_c = cv2.arcLength(circulito, True)**2
                fdf = a/per_c
                if   1/fdf < 15:
                    circulitos_dado += 1
            print(circulitos_dado)

elpapa = cv2.drawContours(f.copy(), contolno_dibujito, -1, (255,0,0), 2)
plt.imshow(elpapa),plt.show()


# ----------------------------------------------------------------




auto = cv2.imread('img01.png')

auto_rgb = cv2.cvtColor(auto, cv2.COLOR_BGR2RGB)

auto_gris = cv2.cvtColor(auto_rgb, cv2.COLOR_RGB2GRAY)

plt.imshow(auto_gris, cmap='gray'), plt.show()




kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
auto_grad = cv2.morphologyEx(auto_gris, cv2.MORPH_GRADIENT, kernel)
plt.imshow(auto_grad, cmap='gray'), plt.show()

auto_resta = cv2.subtract(auto_grad,auto_gris)

plt.imshow(auto_resta, cmap='gray'), plt.show()

tr, auto_resta_bin = cv2.threshold(auto_resta, 120, 255, cv2.THRESH_OTSU)

plt.imshow(auto_resta_bin, cmap='gray'), plt.show()



auto_gris_blur = cv2.GaussianBlur(auto_grad, (11,11), 0)

plt.imshow(auto_gris_blur, cmap='gray'), plt.show()

auto_gris_canny = cv2.Canny(auto_gris_blur, 50, 150)

plt.imshow(auto_gris_canny, cmap='gray'), plt.show()

tr, auto_gris_bin = cv2.threshold(auto_gris_blur, 120, 255, cv2.THRESH_OTSU)

plt.imshow(auto_gris_bin, cmap='gray'), plt.show()

auto_reverse = cv2.bitwise_not(auto_gris_bin)

plt.imshow(auto_reverse, cmap='gray'), plt.show()






