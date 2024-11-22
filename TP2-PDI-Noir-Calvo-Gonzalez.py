import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detectar_monedas_y_dados(monedas: cv2.typing.MatLike)-> str:
    """
    Función que busca y diferencia entre los tipos de monedas de 1 peso, 50 centavos y 10 centavos, devolviendo la cantidad de cada uno

    También, busca y diferencia la cantidad de puntos en la cara superior de distintos dados, devolviendo la cantidad de puntos de cada dado

    Y Muestra una imagen con los contornos encontrados de los dados y las monedas 
    ----------------------------------------------------------------------------------------------------------------------------------------
    Parámetros

    imagen: una imagen que preferentemente contenga monedas de los valores anteriormente dichos y dados  
    """

    monedas_copy = monedas.copy()

    # Imagen donde se van a mostrar los contornos de las monedas y los dados
    moenda_contornos = cv2.cvtColor(monedas.copy(), cv2.COLOR_BGR2RGB)

    # Se suaviza la imagen para obtener los detalles de los objetos de manera más clara
    monedas_blur = cv2.blur(monedas, (5,5))

    # Se convierte de BGR a HLS para trabajar con el canal de la saturación
    monedas_hls = cv2.cvtColor(monedas_blur, cv2.COLOR_BGR2HLS)

    # Separamos los canales para quedarnos con el canal S
    _, _, s = cv2.split(monedas_hls)

    # Se vuelve a suavizar la imagen, pero ahora sobre la saturación
    monedas_hls_s = cv2.blur(s, (7, 7))

    # Se binariza para obtener los objetos y poder trabajar con ellos
    _ , monedas_s_binarizado = cv2.threshold(monedas_hls_s, 14, 255, cv2.THRESH_BINARY)

    # Se realiza una clausura para poder cerrar los huecos de los diferentes objetos
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
    monedas_clausura = cv2.morphologyEx(monedas_s_binarizado, cv2.MORPH_CLOSE, B)

    # Se realiza apertura para poder eliminar ruido que quede en la imagen y, además, poder redondear los objetos para, así, poder obtener sus formas de manera más clara
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))
    monedas_apertura = cv2.morphologyEx(monedas_clausura, cv2.MORPH_OPEN, B)

    # Se buscan los contornos
    contornos_objetos, _ = cv2.findContours(monedas_apertura, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Se crea una imagen para así luego poder dibujar los objetos sobrantes
    mascara_zeros = np.zeros_like(monedas_apertura, dtype=np.uint8)

    contornos_monedas_areas  = []
    contornos_monedas_solos = []

    # Se itera sobre los contornos de cada objeto, obteniendo su factor de forma
    # Se busca poder diferenciar las monedas, que son circulares, de los dados, que son objetos con formas cercanas a cuadrados
    # Guardando el área y contorno del dado
    # Por otra parte, los objetos que no sean circulares, es decir, los dados, se dibujan una máscara y se rellenan para su posterior análisis.
    for contorno in contornos_objetos:
        area_objetos = cv2.contourArea(contorno)
        perimetro_cuadrado_objetos = cv2.arcLength(contorno, True)**2
        factor_de_forma = area_objetos/perimetro_cuadrado_objetos
        if factor_de_forma > 0.062:
            contornos_monedas_areas.append((contorno, area_objetos))
            contornos_monedas_solos.append(contorno)

            # Se relaiza el contorno de las monedas
            cv2.drawContours(moenda_contornos, contorno, -1, (255,0,0),4)
        else:
            cv2.drawContours(mascara_zeros, [contorno], -1, 255, thickness=cv2.FILLED)

    # Se aplica una clausura con un kernel grande, para rellenar los posibles huecos entre los dados, generados por los puntos en los mismos
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 150))
    dado_clausura = cv2.morphologyEx(mascara_zeros, cv2.MORPH_CLOSE, B)

    # Se aplica la apertura para redondear las esquinas y picos de los dados
    B = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))
    dado_apertura = cv2.morphologyEx(dado_clausura, cv2.MORPH_OPEN, B)

    # Se buscan los contornos
    dados_contornos, _ = cv2.findContours(dado_apertura, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(moenda_contornos, dados_contornos, -1, (0,255,0),4)
    moneda_copia_dados = monedas.copy()

    lista_cantidad_dados_puntos = []

    # Se itera para cada dado
    for dado in dados_contornos:
        x, y, w, h = cv2.boundingRect(dado)

        # Se relaiza el contorno cuadrado de los dados
        cv2.rectangle(moenda_contornos, (x-5,y-5),(x+w+5,y+h+5),(0,255,0),4)
        # Se recorta la de la imagen original, el área que fue recortada usando boundingbox, agregando un pequeño margen de error
        dado_recortado = moneda_copia_dados[y-10:y+h+10, x-10:x+w+10]

        dado_gris = cv2.cvtColor(dado_recortado, cv2.COLOR_BGR2GRAY)

        # Con la imagen pasada a escala de grises, se busca suavizarla y binarizarla para así poder obtener los puntos dentro de ella
        dado_blur = cv2.GaussianBlur(dado_gris, (9, 9), 2)
       
        _, dado_binarizado = cv2.threshold(dado_blur, 170,255, cv2.THRESH_BINARY)
        
        # Se invierte la imagen, ya que para buscar los contornos de los dados, necesitamos que estos puntos sean blancos
        dado_binarizado_invertido = cv2.bitwise_not(dado_binarizado)
        dados_contornos_puntos, _ = cv2.findContours(dado_binarizado_invertido, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Se itera sobre cada punto o forma que fue encontrada en el dado, ya que al invertir la imagen, el fondo también es encontrado por find countorns
        # Se buscan las formas que sean circulares, al igual que se realizó anteriormente para las monedas, y se suma 1 por cada punto encontrado
        cantidad_de_puntos_en_dado = 0
        for puntos_y_formas in dados_contornos_puntos:
            area_puntos = cv2.contourArea(puntos_y_formas)
            perimetro_cuadrado_puntos_dados = cv2.arcLength(puntos_y_formas, True)**2
            factor_de_forma_puntos_dado = area_puntos/perimetro_cuadrado_puntos_dados
            if factor_de_forma_puntos_dado > 0.062:
                cantidad_de_puntos_en_dado+=1 

        lista_cantidad_dados_puntos.append(cantidad_de_puntos_en_dado)
    plt.imshow(moenda_contornos), plt.show()
    #-------------------------------------------------------------------------------------------------------------------------------------

    # Proceso para encontrar y diferenciar las monedas
    

    monedas_rgb_copia = cv2.cvtColor(monedas_copy, cv2.COLOR_BGR2RGB)

    # Se ordena el área de las monedas de forma ascendente para facilitar la diferenciación entre ellas 
    contornos_monedas_ordenado = sorted(contornos_monedas_areas , key=lambda x: x[1])

    max_area = contornos_monedas_ordenado[-1][1]
    min_area = contornos_monedas_ordenado[0][1]

    # En este caso, se busca saber si la moneda con menor área es diferente a la de mayor
    # Si esto es verdad, estamos en el caso donde puede llegar a haber 2 o 3 tipos de monedas en la imagen 
    # ya que si el área entre la menor y la mayor es mayor a 0.85, esto quiere decir que, dentro del cierto margen, son el mismo tipo de moneda
    if min_area / max_area < 0.85:
        monedas_10_50 = 0
        monedas_10_peso = 0
        monedas_50 = 0
        monedas_peso = 0

        # Rango de relaciones de área entre monedas
        rel_50_10_max , rel_50_10_min = 0.61, 0.4
        rel_50_1_max , rel_50_1_min = 0.83, 0.74
        rel_1_10_max , rel_1_10_min = 0.73, 0.62

        # Se clasifican las monedas según su relación de área con la moneda más grande, siendo estas relaciones declaradas anteriormente
        for moneda, area in contornos_monedas_areas:
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

    # En el caso de que tengamos 1 solo tipo de moneda
    else:
        monedas_10_peso = 0
        monedas_50 = 0
        monedas_peso = 0
        # Rango de proporciones de área de las monedas de 50 centavos
        rel_recuadro_50_max , rel_recuadro_50_min = 0.97, 0.65

        # Se trabaja sobre la mayor moneda, ya que llegado a este punto, todas las monedas son iguales y la mayor suele ser la que presenta una mejor y más precisa forma
        contorno_moneda = contornos_monedas_ordenado[-1][0]
        x, y, w, h = cv2.boundingRect(contorno_moneda)

        # Se recorta la imagen y se convierte en espacio CIElab
        moneda_recortada = monedas_rgb_copia[y-10:y+h, x:x+w]
        moneda_recortada_cielab = cv2.cvtColor(moneda_recortada, cv2.COLOR_BGR2LAB)

        # Se separa al canal b, ya que para CIElab el canal b, no está tan afectado por la iluminación y específicamente el b se centra en los colores amarillos
        _, _, b_moneda = cv2.split(moneda_recortada_cielab)
        _, b_moneda_binarizada = cv2.threshold(b_moneda, 115, 255, cv2.THRESH_BINARY)

        b_moneda_binarizada_invertida = np.bitwise_not(b_moneda_binarizada)

        # Se aplica la clausura, apertura y clausura nuevamente para redondear y sacar ruido ante cualquier tipo de problema y deformación de las monedas
        B_monedita = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
        b_moneda_binarizada_invertida_clausura = cv2.morphologyEx(b_moneda_binarizada_invertida, cv2.MORPH_CLOSE, B_monedita)

        B_monedita = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        b_moneda_binarizada_invertida_apertura = cv2.morphologyEx(b_moneda_binarizada_invertida_clausura, cv2.MORPH_OPEN, B_monedita)

        B_monedita = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
        b_moneda_clausura_final = cv2.morphologyEx(b_moneda_binarizada_invertida_apertura, cv2.MORPH_CLOSE, B_monedita)

        contorno_moneda_individual, _ = cv2.findContours(b_moneda_clausura_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Si el contorno es 0, eso quiere decir que la moneda no tenía colores amarillos o dorados
        # Por lo tanto, la imagen resultante va a ser totalmente negra, esto quiere decir que es la moneda de 10 centavos
        # ya que find_countorns no es capaz de encontrar ningún valor
        if len(contorno_moneda_individual) == 0:
            monedas_10_peso = len(contornos_monedas_solos)
        else:
            # Para las otras monedas se busca el área resultante del fondo de las monedas de 50 y 1 peso 
            # Resultando en que si el área está dentro de la relación para la moneda de 50
            # Eso quiere decir que es la moneda de un 50
            # En contra parte, para la moneda de 1 peso, su borde dorado reduce el área de la moneda, ampliando por contra parte el área del fondo
            area_moneda_50_o_peso = cv2.contourArea(contorno_moneda_individual[0])
            area_recuadro_moneda_50_o_peso = w*h
            if rel_recuadro_50_min < area_moneda_50_o_peso/area_recuadro_moneda_50_o_peso < rel_recuadro_50_max:
                monedas_50 = len(contornos_monedas_solos)
            else:
                monedas_peso = len(contornos_monedas_solos)

    return print(
        f'Se encontraron: "{monedas_50}" monedas de 50 centavos, "{monedas_peso}" monedas de 1 peso y "{monedas_10_peso}" monedas de 10 centavos\n'
        f'Y tambien se encontro "{len(lista_cantidad_dados_puntos)}" dados y sus respectivos valores son:\n' + "\n".join(f"  - En el dado número {i + 1}, tiene {puntos} puntos" for i, puntos in enumerate(lista_cantidad_dados_puntos)))


def mostrar_patentes(autos: list[cv2.typing.MatLike])-> None:
    """
    Procesa una lista de imágenes de autos, detecta las patentes en cada imagen, las resalta junto con los caracteres internos y las muestra.

    Parameters
    ----------
    autos : list[cv2.typing.MatLike] -> Lista de imágenes (matrices) de autos a procesar

    """

    def encontrar_patente(img_placa: cv2.typing.MatLike, img_letras: cv2.typing.MatLike, img_rgb: cv2.typing.MatLike) -> cv2.typing.MatLike | str:
        """
        Segmenta la patente de un auto y sus letras interiores, mostrandolo en una imagen.

        Parameters
        ----------
        img_placa : Imagen en donde se va a detectar la parte interior de color negro de la patente.

        img_letras : Imagen en donde se va a detectar las letras de la patente.

        img_rgb : Imagen en donde se va a segmentar la patente y las letras (RGB).
        
        Returns
        ----------
        patente_segmentada : Devuelve la imagen rgb ingresada en los parámetros pero con las patentes segmentadas.

        str : Devuelve la string "Mal" en caso de no encontrar la patente.
        """

        # Se buscan, dentro de la imagen binaria, los componentes conectados
        num_labels,_,stats,_ = cv2.connectedComponentsWithStats(img_placa, connectivity=8)
        for componente in range(1,num_labels):
            x,y,w,h,_ = stats[componente]

            # Se validan las posibles proposiciones de la patente (formas rectangulares) y luego el posible área que esta patente pueda tener
            if 1.8 < w/h < 4.2:
                if 300 < w*h < 5000:
                    recorte_placa = img_letras[y:y+h,x:x+w]
                    letras = []

                    # Se buscan los posibles caracteres dentro de la patente con componentes conectados
                    num_labels_placa,_,stats_letras,_ = cv2.connectedComponentsWithStats(recorte_placa, connectivity=8)
                    for num_label in range(1,num_labels_placa):
                        x_l, y_l, w_l, h_l, _ = stats_letras[num_label]

                        # Se validan las posibles proporciones de los caracteres (forma rectangular, verticalmente o casi cuadrada en casos específicos)
                        # Luego se delimita un área mínima para descartar posible ruido como un carácter
                        # Al final se agrega a una lista para así luego poder verificar que estos son los caracteres de una patente 
                        if 0.37 <= w_l/h_l <= 0.9:
                            if 30 <= w_l*h_l:
                                letras.append(num_label)
                    
                    # En el caso donde se encontraron exactamente 6 caracteres, se considera como una patente válida
                    if len(letras) == 6:
                        # Se dibuja una línea sobre el contorno de la patente agregando cierto margen de error
                        patente_segmentada = cv2.rectangle(img_rgb,(x-5,y-5),(x+w+5,y+h+5), (255,0,0), 1)

                        # Se itera por cada carácter y se le dibuja el contorno
                        for letra in letras:
                            x_l, y_l, w_l, h_l, _ = stats_letras[letra]
                            patente_segmentada = cv2.rectangle(img_rgb,(x+x_l,y+y_l),(x+x_l+w_l,y+y_l+h_l), (255,0,0), 1)
                        return patente_segmentada
                        
        return 'Mal'


    for auto in autos:
        
        # Se convierte la imagen a escala de grises y RGB
        auto_rgb = cv2.cvtColor(auto, cv2.COLOR_BGR2RGB)
        auto_gris = cv2.cvtColor(auto, cv2.COLOR_BGR2GRAY)

        # Se aplican diferentes umbrales para segmentar las diferentes áreas de interés
        threshold_bajo = cv2.threshold(auto_gris,112, 255, cv2.THRESH_BINARY)[1]
        threshold_adaptive = cv2.adaptiveThreshold(auto_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)
        threshold_bajo_adpt = cv2.bitwise_and(threshold_adaptive,threshold_adaptive, mask=threshold_bajo)

        threshold_otsu = cv2.threshold(auto_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        threshold_alto = cv2.threshold(auto_gris,145, 255, cv2.THRESH_BINARY)[1]
        threshold_alto_otsu = cv2.bitwise_or(threshold_otsu,threshold_otsu, mask=threshold_alto)

        threshold_letras = cv2.bitwise_or(threshold_alto_otsu,threshold_bajo_adpt)

        # Se realiza una operación de blackhat para así poder resaltar la patente
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(30,15))
        blackhat_img = cv2.morphologyEx(auto_gris,cv2.MORPH_BLACKHAT, kernel)
        blackhat_patente = cv2.subtract(auto_gris,blackhat_img)
        bin_blackhat_patente = np.array(blackhat_patente<1, dtype=np.uint8)*255

        # Se busca detectar la patente
        patente = encontrar_patente(bin_blackhat_patente, threshold_letras, auto_rgb)
        
        # Si se encuentra la patente, esta es devuelta con los marcos a la patente y caracteres ya pintados 
        if type(patente) != str:
            plt.imshow(patente), plt.show()
            pass

        # En el caso donde estos parámetros de umbralizacion y demás no funcionen, se prueba a realizar con diferentes parámetros
        else:
            blackhat_eq = cv2.equalizeHist(auto_gris)

            kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT,(25,5))

            blackhat_img_2 = cv2.morphologyEx(blackhat_eq,cv2.MORPH_BLACKHAT, kernel_2)

            bin_blackhat = cv2.threshold(blackhat_img_2, 100, 255, cv2.THRESH_BINARY)[1]

            bin_blackhat_rev = cv2.bitwise_not(bin_blackhat)

            patente = encontrar_patente(bin_blackhat, bin_blackhat_rev, auto_rgb)
            if type(patente) != str:
                plt.imshow(patente), plt.show()
                pass


autos = []

monedas = cv2.imread('monedas.jpg')

num_elementos = len(os.listdir('./patentes'))

for i in range(1,num_elementos+1):
    autos.append(cv2.imread(f'./patentes/img{i}.png'))


# Menu

while True:
    print('\n1: Ver problema de monedas y dados.\n\n2: Ver patentes\n\n3: Salir del código\n')
    opcion = int(input("Ingrese su opción: "))
    match opcion:
        case 1:
            detectar_monedas_y_dados(monedas)

        case 2:
            mostrar_patentes(autos)

        case 3:
            break