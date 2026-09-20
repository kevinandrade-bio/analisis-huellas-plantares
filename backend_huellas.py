# ==================================================
# IMPORTACIONES
# ==================================================
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening, remove_small_objects, disk # Agrego las herramientas morfológicas
from sklearn.decomposition import PCA
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import cv2


# ==================================================
# FUNCIONES DE PROCESAMIENTO
# ==================================================

# Procesa una imagen de huella: elimina píxeles con colores similares a la huella original,
# la convierte a escala de grises, calcula un umbral automático (Otsu), segmenta la huella
# del fondo y aplica operaciones morfológicas para suavizar contornos y eliminar ruido,
# devolviendo una imagen binaria de la huella y el umbral utilizado.
def procesar_huella(imagen_rgb):

    # 1. Reemplazar los píxeles cercanos a la huella por el color del fondo
    b_min, b_max = 152, 177     #Busque el rango de valores de cada componente de esos pixeles
    g_min, g_max = 97, 145
    r_min, r_max = 112, 127

    nuevo_color = [135, 77, 124]    #Valor de los pixeles del fondo morado

    condicion_r = (imagen_rgb[:, :, 0] >= r_min) & (imagen_rgb[:, :, 0] <= r_max)
    condicion_g = (imagen_rgb[:, :, 1] >= g_min) & (imagen_rgb[:, :, 1] <= g_max)
    condicion_b = (imagen_rgb[:, :, 2] >= b_min) & (imagen_rgb[:, :, 2] <= b_max)

    mascara_total = condicion_r & condicion_g & condicion_b #Todos los pixeles que cumplan las 3 condiciones
    imagen_rgb = imagen_rgb.copy()
    imagen_rgb[mascara_total] = nuevo_color #Los reemplazo por el color morado del fondo

    # 2. Convertir la imagen a escala de grises
    imagen_modificada = Image.fromarray(imagen_rgb) #Agarro la matriz de la imagen modificada y la convierto nuevamente en un objeto imagen
    imagen_gris = imagen_modificada.convert("L") #Convierte la imagen a escala de grises (el modo "L" en Pillow significa Luminance, que representa grises)
    matriz_gris = np.array(imagen_gris) #Vuelvo a convertir el objeto imagen a matriz de numeros, ahora solo tiene un valor entre 0 (negro) y 255 (blanco)

    # 3. Calculo el umbral automaticamente (Metodo Otsu)
    umbral_automatico = threshold_otsu(matriz_gris) #Analiza el histograma (cuantos pixeles hay de cada tono de gris) y calcula el punto de corte para separar en dos grupos definidos (fondo y huella)

    # 4. Segmentación (binarizacion): Separo la huella del fondo usando el umbral calculado
    mascara_huella = matriz_gris > umbral_automatico #Evalua cada valor de la matriz y se fija si es mayor que el umbral. Asigna true o false

    # 5. Morfología (suavizado de contornos)
    elemento_estructurante = disk(9) # Uso el disco morfológico para "redondear" y eliminar los artefactos rectos.
    mascara_suavizada = binary_opening(mascara_huella,elemento_estructurante) #Apertura: Suaviza contornos y elimina artefactos externos delgados. Esto es erosion + dilatacion
    mascara_final = remove_small_objects(mascara_suavizada,min_size=1500) #Eliminación por área: Borra manchas o islas blancas desconectadas.'min_size' indica píxeles mínimos para NO ser borrada.

    # 6. Imagen: Pinto la huella suavizada de blanco (255)
    imagen_segmentada = np.zeros_like(matriz_gris) #Creo una matriz del mismo tamaño que la de mi imagen pero con 0
    imagen_segmentada[mascara_final] = 255 #Uso mi mascara y en la posicion que encuentra un true, le asigno 255 a mi matriz de 0

    return imagen_segmentada, umbral_automatico

# Calcula mediante PCA la orientación principal de la huella segmentada
# y dibuja sobre la imagen el eje de mayor variación junto con su centro.
def dibujar_eje_pca(imagen_segmentada, titulo): #Objetivo es usar PCA (analisis de componentes principales) para encontrar hacia donde apunta el pie

    # Obtener coordenadas de los píxeles blancos
    coords = np.column_stack(np.where(imagen_segmentada > 0)) #Busca los pixeles blancos y quenera una matriz true-false. np.where devuelve las posiciones de los true y no.column las junta en pares ordenados (y,x)

    # Aplicar PCA
    pca = PCA(n_components=2) #Inicializa el algoritmo diciendo que queremos encontrar los 2 ejes principales (largo y ancho)
    pca.fit(coords) #Analiza mi lista de coordenadas y calcula la dereccion de los ejes

    # Centro de la nube de puntos
    centro = np.mean(coords, axis=0) #Calcula el promedio de las coords Y y X, el resultado es el centro de masa de la huella

    # Dirección principal
    vector_principal = pca.components_[0] #Guarda los vectores de los ejes que encontró, el primero es el eje con mas variacion (linea que recorre desde el talon a los dedos). El [1] es el ancho del pie, que no lo uso.

    # Longitud visual de la línea que voy a dibujar
    largo = 300

    # Calcular extremos de la línea
    punto1 = centro - vector_principal * largo #Me paro en el centro y me muevo 300 pixeles hacia atras
    punto2 = centro + vector_principal * largo # y hacia adelante

    # Graficar
    plt.figure(figsize=(6,6))
    plt.imshow(imagen_segmentada, cmap='gray')
    # Línea PCA
    plt.plot([punto1[1], punto2[1]],[punto1[0], punto2[0]],'r-',linewidth=3) #NumPy maneja las matrices como [Fila, Columna] (es decir, [Y, X]). Pero Matplotlib grafica usando [X, Y]. Por esose invierten las posiciones: punto1[1] son X (columnas) y punto1[0] son Y (filas).
    # Centro
    plt.plot(centro[1],centro[0],'bo')
    plt.title(titulo)
    plt.axis('off')
    plt.show()

    # # Mostrar ángulo de inclinacion de cada huella
    # angulo = np.arctan2(vector_principal[0],vector_principal[1])
    # angulo_grados = np.degrees(angulo)
    # print(f"Ángulo principal ({titulo}): {angulo_grados:.2f} grados")

# Calcula el ángulo de inclinación principal de la huella segmentada
# aplicando PCA sobre sus píxeles y devuelve el resultado en grados.
def calcular_angulo_pca(imagen_segmentada): #Forma simplificada de la funcion anterior, la otra fue para comprobar que se calculaba bien el PCA, ahora lo que me interesa es quedarme solo con el angulo de inclinacion

    # Coordenadas de píxeles blancos
    coords = np.column_stack(np.where(imagen_segmentada > 0))

    # PCA
    pca = PCA(n_components=2)
    pca.fit(coords)

    # Vector principal
    vector_principal = pca.components_[0]

    # Ángulo en radianes
    angulo = np.arctan2(vector_principal[0],vector_principal[1])

    # Convertir a grados
    angulo_grados = np.degrees(angulo)
    if angulo_grados < 0:  # PCA encuentra la línea del eje, pero el vector puede apuntar hacia los dedos o hacia el talón, si apunta al talon da un ángulo negativo. Le sumo 180° para que represente la misma inclinación pero siempre en un rango positivo.
        angulo_grados += 180

    return angulo_grados

# Calcula la rotación necesaria a partir del ángulo principal de la
# huella y genera una nueva imagen donde la huella queda orientada
# verticalmente para facilitar análisis posteriores.
def rotar_huella(imagen_segmentada, angulo_pca):

    # Quiero dejar el eje vertical
    angulo_rotacion = angulo_pca - 90 #Calculo la diferencia entre el angulo de inclinacion y 90° para que la huella quede vertical

    # Rotar imagen
    imagen_rotada = rotate(imagen_segmentada,angle=angulo_rotacion,reshape=True,order=0) #Llamo a la funcion rotate de la libreria scipy.ndimage, y la imagen se rota en el angulo calculado

    return imagen_rotada

# Localiza los extremos de la huella segmentada, calcula el rectángulo
# mínimo que la contiene y genera una nueva imagen recortada, devolviendo
# además las coordenadas de origen del recorte.
def recortar_huella(imagen_segmentada):

    # Coordenadas de píxeles blancos
    coords = np.column_stack(np.where(imagen_segmentada > 0))

    # Límites
    y_min, x_min = coords.min(axis=0) #Busco en todas las coordenadas la fila mas alta y la columna mas a la izquierda
    y_max, x_max = coords.max(axis=0) #Busco en todas las coordenadas la fila mas baja y la columna mas a la derecha

    # Recorte de la imagen en esos limites encontrados
    imagen_recortada = imagen_segmentada[y_min:y_max+1,x_min:x_max+1]

    return imagen_recortada, x_min, y_min

# Convierte la imagen segmentada en una máscara binaria y calcula el
# ancho de la huella a lo largo de toda su longitud, contando los
# píxeles blancos presentes en cada fila.
def calcular_perfil_anchura(imagen_segmentada): #Hago una curva en donde represento el ancho de cada zona del pie

    # Convertir a binaria por seguridad
    mascara = imagen_segmentada > 0

    # Sumar píxeles blancos por fila
    perfil_anchura = np.sum(mascara, axis=1) #Suma los elementos a lo largo del eje 1 (columnas). Se para en la fila 0 y cuenta cuantos true hay horizontalmente y se guarda ese numero.

    return perfil_anchura

# Aplica un filtro gaussiano al perfil de anchura para eliminar
# variaciones bruscas y obtener una curva más suave.
def suavizar_perfil(perfil, sigma=4):

    perfil_suavizado = gaussian_filter1d(perfil,sigma=sigma) #Le aplico un filtro a la curva que encontre anteriormente para eliminar la forma de serrucho poder analizarla luego

    return perfil_suavizado

# Analiza el perfil de anchura suavizado para localizar el valle que
# separa los dedos del resto de la huella. Para ello identifica el pico
# principal del antepié, busca mínimos locales previos y selecciona el
# primero asociado a una región significativamente más angosta que el
# antepié, devolviendo la fila donde debe realizarse el corte.
def detectar_corte_dedos(perfil_suavizado):

    # Encontrar pico principal (antepié)
    indice_antepie = np.argmax(perfil_suavizado) #Busco en que fila de la matriz esta el punto mas alto de la curva. Se que es del antepie
    altura_antepie = perfil_suavizado[indice_antepie] #Guarda el valor de anchura del pico

    # Analizar solo ANTES del antepié
    region_superior = perfil_suavizado[:indice_antepie] #Corto la curva y me quedo solo con la parte que va desde el principio de la imagen hasta el pico del antepié que encontre.

    # Buscar mínimos locales, invierto la señal para encontrar valles
    valles, _ = find_peaks(-region_superior,prominence=3) #find_peaks busca picos (máximos). Para encontrar valles (mínimos), le paso la señal invertida. prominence=5 detecta valles que tengan una caída/subida de al menos 5 píxeles.
    # print(f"Índice antepié: {indice_antepie} (Altura: {altura_antepie:.2f})")

    # Si NO hay valles no hay dedos
    if len(valles) == 0:
        # print("No se detectaron dedos") #Si no hay valles es porque la imagen de la huella no tiene dedos
        
        return 0

    # Filtrar valles de derecha a izquierda
    corte = 0
    
    # reversed(valles) empieza por el valle más cercano al antepié y va hacia la izquierda
    for v in reversed(valles): #Inicio un bucle para revisar los valles encontrados, pero usando reversed(). Empieza a analizar desde el valle más cercano al antepié y avanza hacia atrás (hacia la punta del pie).
        # Encontrar el pico más alto a la izquierda de este valle candidato
        altura_pico_izq = np.max(perfil_suavizado[:v])
        
        # Calculo la relación del pico encontrado respecto al pico del antepié
        relacion = altura_pico_izq / altura_antepie
        
        # Si el pico de la izquierda es notablemente más bajo, ¡son los dedos!
        umbral_relativo=0.75 #Para considerar que un pico pertenece a los dedos, su ancho máxima no debe superar el 75% del ancho del antepié.
        if relacion < umbral_relativo:
            corte = v #Guarda la fila del valle actual v como el punto donde se debe cortar la imagen para separar los dedos.
            # print(f"¡Valle de los dedos real detectado en fila: {corte}!")
            # print(f" -> Altura pico dedos: {altura_pico_izq:.2f} (Relación: {relacion:.2f})")
            break
        else:
            # Si entra aquí, es el falso pico bimodal del antepié
            print(f"Ignorando valle en fila {v}. El pico a su izquierda es muy alto (Relación: {relacion:.2f}), sigue siendo el antepié.")

    if corte == 0:
        print("No se detectó un corte válido (todos los picos previos eran muy altos, posiblemente no hay dedos segmentados)")
        
    return corte

# Separa los dedos del resto de la huella utilizando la fila de corte
# calculada previamente. Después elimina componentes pequeñas aisladas
# generadas por cortes parciales y devuelve una imagen binaria limpia
# que contiene únicamente la región principal de la huella.
def eliminar_dedos(imagen_binaria, fila_corte):

    imagen_sin_dedos = imagen_binaria[fila_corte:, :] #Me quedo con lo que esta desde fila_corte hacia abajo (elimino los dedos)

    # Aseguro tipo booleano
    imagen_sin_dedos = imagen_sin_dedos.astype(bool)

    # Elimino componentes pequeños, porque quedaron algunos dedos a medio cortar, separados de la huella
    imagen_sin_dedos = remove_small_objects(imagen_sin_dedos,min_size=500)

    return (imagen_sin_dedos.astype(np.uint8)*255)

# Busca las líneas de contorno que delimitan la huella en la imagen
# binaria. Si se detectan contornos, devuelve sus coordenadas; en caso
# contrario, devuelve None.
def obtener_contorno(imagen_binaria):

    # contornos = find_contours(imagen_binaria, level=127)

    # if len(contornos) == 0:
    #     return None

    # return contornos
    # Aseguramos que la imagen sea uint8 (OpenCV es estricto con esto)
    img_uint8 = imagen_binaria.astype(np.uint8)
    
    # cv2.RETR_EXTERNAL: Trae solo el contorno de afuera (evita sub-contornos raros)
    # cv2.CHAIN_APPROX_SIMPLE: Comprime los puntos para ahorrar memoria
    contornos, _ = cv2.findContours(img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contornos) == 0:
        return None

    return contornos # Ahora devuelve la lista secuencial perfecta de OpenCV

# ==================================================
# FUNCIONES DE ARCH INDEX
# ==================================================

# Localiza puntos representativos en los extremos superior e inferior de
# la huella para posteriormente calcular el eje longitudinal que los une.
# Para evitar la influencia de irregularidades laterales, el cálculo del extremo
# superior se realiza únicamente sobre la zona central de la huella. Devuelve los 2 
# puntos encontrados.
def puntos_extremos_eje_longitudinal(imagen_binaria):

    # Obtiene las coordenadas (fila, columna) de todos los píxeles pertenecientes a la huella (píxeles blancos > 0)
    coords = np.column_stack(np.where(imagen_binaria > 0))
    # Extrae las dimensiones (alto y ancho) de la matriz de la imagen binaria
    alto, ancho = imagen_binaria.shape
    
    # Zona central de la huella, ignoro cierto % de cada lado
    limite_izq = int(0.32 * ancho)
    limite_der = int(0.72 * ancho)
    # Filtra las coordenadas para conservar únicamente las que están dentro de la zona central definida
    coords_centrales = coords[(coords[:,1] >= limite_izq) & (coords[:,1] <= limite_der)]

    # Punto superior
    y_min = coords_centrales[:,0].min()
    banda_superior = coords_centrales[coords_centrales[:,0] <= y_min + 5]
    x_superior = np.mean(banda_superior[:,1])

    # Punto inferior
    y_max = coords[:,0].max()
    banda_inferior = coords[coords[:,0] >= y_max - 5]
    x_inferior = np.mean(banda_inferior[:,1])

    punto_superior = (x_superior, y_min)
    punto_inferior = (x_inferior, y_max)

    return punto_superior, punto_inferior

# Utiliza el eje longitudinal definido por los extremos de la huella
# para ubicar los puntos situados al 33% y 66% de su longitud, y devuelve
# las líneas perpendiculares que delimitan las tres regiones del
# Arch Index (antepié, mediopié y retropié).
def calcular_divisiones_arch_index(punto_superior,punto_inferior,longitud_linea=400):

    x1, y1 = punto_superior
    x2, y2 = punto_inferior

    # Vector longitudinal
    vx = x2 - x1
    vy = y2 - y1

    longitud = np.sqrt(vx**2 + vy**2)

    # Unitario longitudinal
    ux = vx / longitud
    uy = vy / longitud

    # Puntos al 33% y 66%
    p1_x = x1 + vx/3
    p1_y = y1 + vy/3

    p2_x = x1 + 2*vx/3
    p2_y = y1 + 2*vy/3

    # Vector perpendicular unitario
    perp_x = -uy
    perp_y = ux

    mitad = longitud_linea / 2

    linea1 = ((p1_x - mitad*perp_x, p1_y - mitad*perp_y),(p1_x + mitad*perp_x, p1_y + mitad*perp_y))
    linea2 = ((p2_x - mitad*perp_x, p2_y - mitad*perp_y),(p2_x + mitad*perp_x, p2_y + mitad*perp_y))

    return linea1, linea2

# Recorre todos los píxeles pertenecientes a la huella y determina su
# ubicación relativa sobre el eje longitudinal. Según esa posición, los
# clasifica en antepié, mediopié o retropié y calcula el área de cada
# región como cantidad de píxeles que la componen.
def calcular_areas_arch_index(imagen_binaria,punto_superior,punto_inferior):

    mascara = imagen_binaria > 0

    ys, xs = np.where(mascara)

    x1, y1 = punto_superior
    x2, y2 = punto_inferior

    vx = x2 - x1
    vy = y2 - y1

    longitud2 = vx**2 + vy**2

    area_antepie = 0
    area_mediopie = 0
    area_retropie = 0

    for x, y in zip(xs, ys):
        t = ((x - x1)*vx +(y - y1)*vy) / longitud2
        if t < 1/3:
            area_antepie += 1
        elif t < 2/3:
            area_mediopie += 1
        else:
            area_retropie += 1

    return (area_antepie,area_mediopie,area_retropie)

# Clasifica los píxeles de la huella en antepié, mediopié y retropié
# utilizando la división en tercios definida sobre el eje longitudinal,
# y crea una imagen RGB donde cada región se representa con un color
# diferente para facilitar su visualización.
def colorear_regiones_arch_index(imagen_binaria,punto_superior,punto_inferior):

    mascara = imagen_binaria > 0
    ys, xs = np.where(mascara)
    x1, y1 = punto_superior
    x2, y2 = punto_inferior
    vx = x2 - x1
    vy = y2 - y1
    longitud2 = vx**2 + vy**2
    imagen_color = np.zeros((imagen_binaria.shape[0],imagen_binaria.shape[1],3),dtype=np.uint8)

    for x, y in zip(xs, ys):
        t = ((x - x1)*vx +(y - y1)*vy) / longitud2
        if t < 1/3:
            imagen_color[y, x] = [100, 149, 237]      # Rojo
        elif t < 2/3:
            imagen_color[y, x] = [119, 198, 110]      # Verde
        else:
            imagen_color[y, x] = [230, 126, 114]      # Azul

    return imagen_color


# ==================================================
# FUNCIONES DE CSI Y STAHELI
# ==================================================

# Analiza las regiones de antepié y retropié para localizar los puntos
# más mediales de la huella. La selección depende de si se trata de un
# pie derecho o izquierdo, y los puntos obtenidos permiten construir
# posteriormente el eje medial del pie.
def puntos_eje_medial(imagen_binaria,punto_superior,punto_inferior,lado):

    mascara = imagen_binaria > 0
    ys, xs = np.where(mascara)
    x1, y1 = punto_superior
    x2, y2 = punto_inferior
    vx = x2 - x1
    vy = y2 - y1
    longitud2 = vx**2 + vy**2
    antepie = []
    retropie = []

    for x, y in zip(xs, ys):
        t = ((x - x1)*vx +(y - y1)*vy) / longitud2
        fila_metatarsos = punto_superior[1]
        if t < 1/3 and y >= fila_metatarsos: #Detecta debajo de la altura del punto que encontro en el metatarso, porque algunas imagenes todavia tienen el dedo gordo y puede detectar el punto medial en esa zona
            antepie.append((x, y))
        elif t > 2/3:
            retropie.append((x, y))
    antepie = np.array(antepie)
    retropie = np.array(retropie)

    # Pie izquierdo: medial = x máximo
    if lado == "izquierdo":
        indice_antepie = np.argmax(antepie[:,0])
        indice_retropie = np.argmax(retropie[:,0])
    # Pie derecho: medial = x mínimo
    else:
        indice_antepie = np.argmin(antepie[:,0])
        indice_retropie = np.argmin(retropie[:,0])

    punto_medial_antepie = tuple(antepie[indice_antepie])
    punto_medial_retropie = tuple(retropie[indice_retropie])

    return (punto_medial_antepie,punto_medial_retropie)

# Extiende el eje medial hasta los límites superior e inferior de la
# huella y devuelve las coordenadas de sus nuevos extremos.
def extender_eje_medial(medial_ante,medial_retro,punto_superior,punto_inferior):

    x1, y1 = medial_ante
    x2, y2 = medial_retro

    # Vector del eje medial
    vx = x2 - x1
    vy = y2 - y1

    # Evitar división por cero
    if abs(vy) < 1e-6:
        return medial_ante, medial_retro

    # Fila superior e inferior anatómicas
    y_sup = punto_superior[1]
    y_inf = punto_inferior[1]

    # Interpolar sobre la recta medial
    t_sup = (y_sup - y1) / vy
    t_inf = (y_inf - y1) / vy

    punto_sup_ext = (x1 + t_sup * vx,y_sup)
    punto_inf_ext = (x1 + t_inf * vx,y_inf)

    return punto_sup_ext, punto_inf_ext

# Calcula el vector unitario asociado al eje medial de la huella y
# determina su dirección perpendicular, que puede utilizarse para
# construir líneas transversales o realizar mediciones geométricas.
def obtener_perpendicular_eje_medial(punto_antepie,punto_retropie):

    x1, y1 = punto_antepie
    x2, y2 = punto_retropie
    vx = x2 - x1
    vy = y2 - y1
    longitud = np.sqrt(vx**2 + vy**2)
    ux = vx / longitud
    uy = vy / longitud

    # perpendicular unitario
    perp_x = -uy
    perp_y = ux

    return perp_x, perp_y

# Utiliza los puntos que definen el eje medial para generar un conjunto
# de secciones perpendiculares distribuidas regularmente a lo largo de
# dicho eje. Cada sección se representa mediante sus puntos inicial y
# final, y puede emplearse para realizar análisis geométricos de la huella.
def generar_secciones_mediales(punto_antepie,punto_retropie,cantidad=75,largo=400):

    x1, y1 = punto_antepie
    x2, y2 = punto_retropie

    vx = x2 - x1
    vy = y2 - y1

    longitud = np.sqrt(vx**2 + vy**2)

    ux = vx / longitud
    uy = vy / longitud

    perp_x = -uy
    perp_y = ux

    mitad = largo / 2

    secciones = []

    for t in np.linspace(0, 1, cantidad):

        xc = x1 + t * vx
        yc = y1 + t * vy

        x_ini = xc - mitad * perp_x
        y_ini = yc - mitad * perp_y

        x_fin = xc + mitad * perp_x
        y_fin = yc + mitad * perp_y

        secciones.append({"t": t,"inicio": (x_ini, y_ini),"fin": (x_fin, y_fin)})
    
    return secciones

# def generar_secciones_mediales(punto_antepie, punto_retropie, lado, cantidad=80, largo=210):
#     x1, y1 = punto_antepie
#     x2, y2 = punto_retropie

#     # Vector director del eje medial
#     vx = x2 - x1
#     vy = y2 - y1

#     longitud_eje = np.sqrt(vx**2 + vy**2)

#     # Vector unitario director
#     ux = vx / longitud_eje
#     uy = vy / longitud_eje

#     # Vector unitario perpendicular (original, hacia la izquierda si miras de retropie a antepie)
#     perp_x = -uy
#     perp_y = ux

#     # Determinar el factor de dirección para el lado lateral
#     # Para el pie izquierdo, la dirección perpendicular original apunta hacia el lateral.
#     # Para el pie derecho, el lateral está en la dirección opuesta a la perpendicular original.
#     if lado == "izquierdo":
#         factor_direccion = 1  # Misma dirección que perp_x, perp_y
#     elif lado == "derecho":
#         factor_direccion = -1 # Dirección opuesta a perp_x, perp_y
#     else:
#         raise ValueError("El parámetro 'lado' debe ser 'izquierdo' o 'derecho'")

#     secciones = []

#     # Generar secciones a lo largo del eje
#     for t in np.linspace(0, 1, cantidad):
#         # Punto central (xc, yc) sobre el eje medial, será el inicio de la sección
#         xc = x1 + t * vx
#         yc = y1 + t * vy

#         # El punto de inicio de la sección gráfica es el punto sobre el eje
#         x_ini = xc
#         y_ini = yc

#         # El punto final se extiende 'largo' en la dirección lateral determinada
#         # Usamos 'largo' completo, no la mitad, y lo multiplicamos por el factor de dirección.
#         x_fin = xc + (largo * factor_direccion * perp_x)
#         y_fin = yc + (largo * factor_direccion * perp_y)

#         secciones.append({"t": t, "inicio": (x_ini, y_ini), "fin": (x_fin, y_fin)})

#     return secciones

# Muestrea una línea de sección transversal sobre la huella para localizar
# los puntos donde esta intersecta la región segmentada. A partir del
# primer y último punto detectados dentro de la huella, calcula el ancho
# correspondiente y devuelve tanto la distancia medida como los extremos
# utilizados para el cálculo.
def medir_ancho_seccion(imagen_binaria,punto_inicio,punto_fin,n_muestras=800):

    xs = np.linspace(punto_inicio[0],punto_fin[0],n_muestras)
    ys = np.linspace(punto_inicio[1],punto_fin[1],n_muestras)
    puntos_dentro = []
    alto, ancho = imagen_binaria.shape

    for x, y in zip(xs, ys):
        xi = int(round(x))
        yi = int(round(y))
        if (0 <= xi < ancho and 0 <= yi < alto):
            if imagen_binaria[yi, xi] > 0:
                puntos_dentro.append((x, y))
    if len(puntos_dentro) < 2:
        return None, None, None
    p1 = puntos_dentro[0]
    p2 = puntos_dentro[-1]
    distancia = np.sqrt((p2[0] - p1[0])**2 +(p2[1] - p1[1])**2)

    return distancia, p1, p2

# Recorre todas las secciones transversales de la huella, mide el ancho
# en cada una de ellas y almacena tanto los valores obtenidos como los
# puntos utilizados para cada medición. Devuelve un vector con las anchuras 
# obtenidas y la información geométrica asociada.
def calcular_perfil_anchuras(imagen_binaria,secciones):
    
    perfil = []
    extremos = []
    
    for seccion in secciones:
        ancho, p1, p2 = medir_ancho_seccion(imagen_binaria,seccion["inicio"],seccion["fin"])
        if ancho is None:
            perfil.append(0)
            extremos.append(None)
        else:
            perfil.append(ancho)
            extremos.append((p1, p2))

    return np.array(perfil), extremos

# Divide el perfil de anchuras en las regiones de antepié, mediopié y
# retropié utilizando las posiciones relativas de las secciones. Luego
# determina los valores característicos de cada región (máximo en
# antepié, mínimo en mediopié y máximo en retropié) y devuelve tanto
# dichos anchos como los índices de las secciones donde fueron encontrados.
def obtener_anchos_indices(perfil,secciones):

    ts = np.array([s["t"] for s in secciones])

    # Antepié
    idx_ante = np.where((ts >= 0) & (ts < 1/3))[0]

    # Mediopié
    idx_medio = np.where((ts >= 1/3) & (ts < 2/3))[0]

    # Retropié
    idx_retro = np.where((ts >= 2/3) & (ts <= 1))[0]

    # print("ts =", ts)
    # print("idx_ante =", idx_ante)
    # print("idx_medio =", idx_medio)
    # print("idx_retro =", idx_retro)
    # print(ts.min())
    # print(ts.max())
    # print(len(ts))
    ancho_max_antepie = np.max(perfil[idx_ante])
    indice_max_antepie = idx_ante[np.argmax(perfil[idx_ante])]
    
    ancho_min_mediopie = np.min(perfil[idx_medio])
    indice_min_mediopie = idx_medio[np.argmin(perfil[idx_medio])]

    ancho_max_retropie = np.max(perfil[idx_retro])
    indice_max_retropie = idx_retro[np.argmax(perfil[idx_retro])]

    return (ancho_max_antepie,ancho_min_mediopie,ancho_max_retropie,indice_max_antepie,indice_min_mediopie,indice_max_retropie)

# Accede a una sección transversal previamente generada y devuelve los
# puntos que definen sus extremos, permitiendo localizar o representar
# gráficamente dicha sección sobre la huella.
def obtener_linea_seccion(secciones, indice):

    seccion = secciones[indice]

    return (seccion["inicio"],seccion["fin"])


# ==================================================
# FUNCIONES DE VISUALIZACIÓN
# ==================================================

# Rota una imagen RGB según el ángulo especificado y ajusta su orientación
# para mantener la correspondencia con las transformaciones aplicadas a la huella.
def rotar_imagen_rgb(imagen_rgb, angulo_rotacion):

    imagen_rotada = rotate(imagen_rgb,angle=angulo_rotacion,axes=(1,0),reshape=True,order=1)
    imagen_rotada = np.transpose(imagen_rotada,(1,0,2))    
    imagen_rotada = np.fliplr(imagen_rotada)

    return imagen_rotada.astype(np.uint8)

# Transforma las coordenadas de un punto obtenidas sobre la huella
# procesada al sistema de referencia de la imagen RGB original. Para
# ello corrige los desplazamientos producidos por el recorte de la
# huella y por la eliminación de la región de los dedos.
def transformar_punto_a_rgb(punto, x_min, y_min, fila_corte):

    x, y = punto
    x_rgb = x + x_min
    y_rgb = y + y_min + fila_corte

    return (x_rgb, y_rgb)

# Transforma las coordenadas de un segmento obtenido sobre la huella
# procesada al sistema de referencia de la imagen RGB rotada.
# Corrige los desplazamientos introducidos por el recorte y por
# la eliminación de los dedos.
def transformar_segmento_a_rgb(segmento, x_min, y_min, fila_corte):

    punto1, punto2 = segmento

    punto1_rgb = transformar_punto_a_rgb(punto1,x_min,y_min,fila_corte)
    punto2_rgb = transformar_punto_a_rgb(punto2,x_min,y_min,fila_corte)

    return (punto1_rgb, punto2_rgb)

# Transforma las coordenadas del contorno obtenido sobre la huella
# procesada al sistema de referencia de la imagen RGB original. Para
# ello corrige los desplazamientos producidos por el recorte de la
# imagen y por la eliminación de la región de los dedos.
def transformar_contorno_a_rgb(contorno, x_min, y_min, fila_corte):

    # contorno_rgb = contorno.copy()

    # contorno_rgb[:,0] = contorno_rgb[:,0] + y_min + fila_corte
    # contorno_rgb[:,1] = contorno_rgb[:,1] + x_min

    # return contorno_rgb
    contorno_rgb = contorno.copy()

    # IMPORTANTE: Como OpenCV te da (X, Y), el orden de los índices se invierte
    # contorno_rgb[:, 0, 0] es la coordenada X
    # contorno_rgb[:, 0, 1] es la coordenada Y
    contorno_rgb[:, 0, 0] = contorno_rgb[:, 0, 0] + x_min
    contorno_rgb[:, 0, 1] = contorno_rgb[:, 0, 1] + y_min + fila_corte

    return contorno_rgb

# Visualiza sobre la imagen RGB los elementos utilizados para el
# cálculo del Arch Index. Dibuja los contornos de la huella, los
# puntos anatómicos que definen el eje longitudinal y las líneas
# que dividen la huella en regiones para el cálculo del índice.
# Además, muestra el valor obtenido del Arch Index en el título.
def graficar_arch_index(imagen_rgb, resultado):

    arch = resultado["arch_index"]
    punto_superior = arch["punto_superior"]
    punto_inferior = arch["punto_inferior"]
    linea1 = arch["linea1"]
    linea2 = arch["linea2"]
    valor_arch_index = arch["valor"]
    contornos = arch["contornos"]

    fig, ax = plt.subplots(figsize=(8,10))
    ax.imshow(imagen_rgb)
    # Contornos
    for c in contornos:
        ax.plot(c[:,1], c[:,0], color='red', linewidth=2)
    # Puntos anatómicos
    ax.scatter(punto_superior[0],punto_superior[1],color='yellow',s=80,label='Punto superior')
    ax.scatter(punto_inferior[0],punto_inferior[1],color='cyan',s=80,label='Punto inferior')
    # Eje longitudinal
    ax.plot([punto_superior[0], punto_inferior[0]],[punto_superior[1], punto_inferior[1]],color='lime',linewidth=2,label='Eje longitudinal')
    # Primera división
    ax.plot([linea1[0][0], linea1[1][0]],[linea1[0][1], linea1[1][1]],color='magenta',linewidth=2)
    # Segunda división
    ax.plot([linea2[0][0], linea2[1][0]],[linea2[0][1], linea2[1][1]],color='magenta',linewidth=2)
    ax.set_title(f"Arch Index = {valor_arch_index:.3f}")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# Visualiza sobre la imagen RGB los elementos utilizados para el
# cálculo del Chippaux-Smirak Index (CSI). Dibuja los contornos
# de la huella, los puntos que definen el eje medial y las
# mediciones de ancho del antepié y del mediopié empleadas en
# el cálculo del índice.    
def graficar_csi(imagen_rgb, resultado):

    csi = resultado["csi"]
    valor = csi["valor"]
    punto_ante = csi["punto_medial_antepie"]
    punto_retro = csi["punto_medial_retropie"]
    eje = csi["eje_medial"]
    ancho_ante = csi["ancho_antepie"]
    ancho_medio = csi["ancho_mediopie"]
    contornos = csi["contornos"]

    fig, ax = plt.subplots(figsize=(8,10))
    ax.imshow(imagen_rgb)
    # Contornos
    for c in contornos:
        ax.plot(c[:,1], c[:,0], color='red', linewidth=2)
    # Puntos mediales
    ax.scatter(punto_ante[0],punto_ante[1],color='yellow',s=80)
    ax.scatter(punto_retro[0],punto_retro[1],color='cyan',s=80)
    # Eje medial
    ax.plot([eje[0][0], eje[1][0]],[eje[0][1], eje[1][1]],color='lime',linewidth=2)
    # Ancho antepié
    ax.plot([ancho_ante[0][0], ancho_ante[1][0]],[ancho_ante[0][1], ancho_ante[1][1]],color='magenta',linewidth=3)
    # Ancho mediopié
    if ancho_medio is not None:
        ax.plot([ancho_medio[0][0], ancho_medio[1][0]],[ancho_medio[0][1], ancho_medio[1][1]],color='yellow',linewidth=3)
    ax.set_title(f"CSI = {valor:.2f}")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# Visualiza sobre la imagen RGB los elementos utilizados para
# el cálculo del índice de Staheli. Dibuja los contornos de la
# huella, los puntos que definen el eje medial y las mediciones
# de ancho del retropié y del mediopié empleadas en el cálculo
# del índice.
def graficar_staheli(imagen_rgb, resultado):

    si = resultado["staheli"]
    valor = si["valor"]
    punto_ante = si["punto_medial_antepie"]
    punto_retro = si["punto_medial_retropie"]
    eje = si["eje_medial"]
    ancho_retro = si["ancho_retropie"]
    ancho_medio = si["ancho_mediopie"]
    contornos = si["contornos"]

    fig, ax = plt.subplots(figsize=(8,10))
    ax.imshow(imagen_rgb)
    # Contornos
    for c in contornos:
        ax.plot(c[:,1], c[:,0], color='red', linewidth=2)
    # Puntos mediales
    ax.scatter(punto_ante[0],punto_ante[1],color='yellow',s=80)
    ax.scatter(punto_retro[0],punto_retro[1],color='cyan',s=80)
    # Eje medial
    ax.plot([eje[0][0], eje[1][0]],[eje[0][1], eje[1][1]],color='lime',linewidth=2)
    # Ancho retropié
    ax.plot([ancho_retro[0][0], ancho_retro[1][0]],[ancho_retro[0][1], ancho_retro[1][1]],color='magenta',linewidth=3)
    # Ancho mediopié
    if ancho_medio is not None:
        ax.plot([ancho_medio[0][0], ancho_medio[1][0]],[ancho_medio[0][1], ancho_medio[1][1]],color='yellow',linewidth=3)
    ax.set_title(f"Staheli = {valor:.2f}")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# ==================================================
# NUEVAS FUNCIONES DE DIBUJO PARA LA GUI (USANDO OPENCV)
# ==================================================

# Renderiza de forma directa sobre la imagen RGB los elementos geométricos
# del Arch Index. Dibuja los contornos externos, el eje longitudinal y
# las divisiones en tercios para ofrecer una respuesta visual rápida en la GUI.
def dibujar_retroalimentacion_ai(imagen_rgb, resultado_ai):
    """Pinta las líneas y puntos del Arch Index directamente en la matriz de la imagen."""
    # Hacemos una copia para no destruir la imagen original por si se recalcula
    img_destino = imagen_rgb.copy()
    ai = resultado_ai["arch_index"]
    
    p_sup = tuple(map(int, ai["punto_superior"]))
    p_inf = tuple(map(int, ai["punto_inferior"]))
    l1_inicio = tuple(map(int, ai["linea1"][0]))
    l1_fin = tuple(map(int, ai["linea1"][1]))
    l2_inicio = tuple(map(int, ai["linea2"][0]))
    l2_fin = tuple(map(int, ai["linea2"][1]))
    
    contornos = ai["contornos"] # Lista de arrays de puntos

    # 2. DIBUJAR LOS CONTORNOS (Línea roja)
    # Al pasarle la lista completa (-1), OpenCV dibuja todo de un solo tiro y sin rombos
    if contornos is not None:
        cv2.drawContours(img_destino, contornos, -1, [80, 80, 80], 3)
    
    # Dibujar Eje Longitudinal (Línea verde) - OpenCV usa BGR, invertimos a RGB si es necesario.
    # Como la GUI espera RGB, [0, 255, 0] es Verde brillante.
    cv2.line(img_destino, p_sup, p_inf, [200, 200, 200], 2)
    
    # Dibujar Divisiones de Tercios (Líneas Magenta/Rosa)
    cv2.line(img_destino, l1_inicio, l1_fin, [200, 200, 200], 3)
    cv2.line(img_destino, l2_inicio, l2_fin, [200, 200, 200], 3)
    
    # Dibujar Puntos Anatómicos (Círculos)
    cv2.circle(img_destino, p_sup, 7, [255, 255, 255], -1) # Relleno blanco
    cv2.circle(img_destino, p_sup, 7, [50, 50, 50], 1)     # Borde gris oscuro
    cv2.circle(img_destino, p_inf, 7, [255, 255, 255], -1) # Relleno blanco
    cv2.circle(img_destino, p_inf, 7, [50, 50, 50], 1)     # Borde gris oscuro
    
    return img_destino

# Renderiza de forma directa sobre la imagen RGB los elementos geométricos
# del Chippaux-Smirak Index (CSI). Dibuja los contornos externos, el eje medial
# y los segmentos transversales del antepié y mediopié para la interfaz.
def dibujar_retroalimentacion_csi(imagen_rgb, resultado_csi):
    """Pinta las líneas y puntos del Chippaux-Smirak directamente en la matriz."""
    img_destino = imagen_rgb.copy()
    csi = resultado_csi["csi"]
    
    p_ante = tuple(map(int, csi["punto_medial_antepie"]))
    p_retro = tuple(map(int, csi["punto_medial_retropie"]))
    eje_ini = tuple(map(int, csi["eje_medial"][0]))
    eje_fin = tuple(map(int, csi["eje_medial"][1]))
    
    contornos = csi["contornos"] # Lista de arrays de puntos

    # 2. DIBUJAR LOS CONTORNOS (Línea roja)
    # Al pasarle la lista completa (-1), OpenCV dibuja todo de un solo tiro y sin rombos
    if contornos is not None:
        cv2.drawContours(img_destino, contornos, -1, [80, 80, 80], 3)
        
    # Eje medial
    cv2.line(img_destino, eje_ini, eje_fin, [200, 200, 200], 2)
    
    # Ancho antepié (Línea magenta)
    if csi["ancho_antepie"] is not None:
        a_ante_ini = tuple(map(int, csi["ancho_antepie"][0]))
        a_ante_fin = tuple(map(int, csi["ancho_antepie"][1]))
        cv2.line(img_destino, a_ante_ini, a_ante_fin, [255, 255, 255], 4)
        
    # Ancho mediopié (Línea amarilla)
    if csi["ancho_mediopie"] is not None:
        a_medio_ini = tuple(map(int, csi["ancho_mediopie"][0]))
        a_medio_fin = tuple(map(int, csi["ancho_mediopie"][1]))
        cv2.line(img_destino, a_medio_ini, a_medio_fin, [255, 255, 255], 4)
        
    # Puntos mediales
    cv2.circle(img_destino, p_ante, 7, [255, 255, 255], -1)
    cv2.circle(img_destino, p_ante, 7, [50, 50, 50], 1)     # Borde gris oscuro
    cv2.circle(img_destino, p_retro, 7, [255, 255, 255], -1)
    cv2.circle(img_destino, p_retro, 7, [50, 50, 50], 1)     # Borde gris oscuro
    
    return img_destino

# Renderiza de forma directa sobre la imagen RGB los elementos geométricos
# del índice de Staheli. Dibuja los contornos externos, el eje medial
# y los segmentos transversales del retropié y mediopié para la interfaz.
def dibujar_retroalimentacion_si(imagen_rgb, resultado_si):
    """Pinta las líneas y puntos del índice de Staheli directamente en la matriz."""
    img_destino = imagen_rgb.copy()
    si = resultado_si["staheli"]
    
    p_ante = tuple(map(int, si["punto_medial_antepie"]))
    p_retro = tuple(map(int, si["punto_medial_retropie"]))
    eje_ini = tuple(map(int, si["eje_medial"][0]))
    eje_fin = tuple(map(int, si["eje_medial"][1]))
    
    contornos = si["contornos"] # Lista de arrays de puntos

    # 2. DIBUJAR LOS CONTORNOS (Línea roja)
    # Al pasarle la lista completa (-1), OpenCV dibuja todo de un solo tiro y sin rombos
    if contornos is not None:
        cv2.drawContours(img_destino, contornos, -1, [80, 80, 80], 3)
        
    # Eje medial
    cv2.line(img_destino, eje_ini, eje_fin, [200, 200, 200], 2)
    
    # Ancho retropié (Línea magenta)
    if si["ancho_retropie"] is not None:
        a_retro_ini = tuple(map(int, si["ancho_retropie"][0]))
        a_retro_fin = tuple(map(int, si["ancho_retropie"][1]))
        cv2.line(img_destino, a_retro_ini, a_retro_fin, [255, 255, 255], 4)
        
    # Ancho mediopié (Línea amarilla)
    if si["ancho_mediopie"] is not None:
        a_medio_ini = tuple(map(int, si["ancho_mediopie"][0]))
        a_medio_fin = tuple(map(int, si["ancho_mediopie"][1]))
        cv2.line(img_destino, a_medio_ini, a_medio_fin, [255, 255, 255], 4)
        
    # Puntos mediales
    cv2.circle(img_destino, p_ante, 7, [255, 255, 255], -1)
    cv2.circle(img_destino, p_ante, 7, [50, 50, 50], 1)     # Borde gris oscuro
    cv2.circle(img_destino, p_retro, 7, [255, 255, 255], -1)
    cv2.circle(img_destino, p_retro, 7, [50, 50, 50], 1)     # Borde gris oscuro
    
    return img_destino

# Recalcula el Arch Index de forma dinámica cuando el usuario desplaza los
# puntos de control en la GUI, mapeando las nuevas coordenadas desde el espacio
# RGB al sistema de la máscara binaria para actualizar áreas y líneas divisorias.
def recalcular_arch_index(resultado_ai, mascara, nuevo_sup_rgb, nuevo_inf_rgb):
    
    # Recalcula el Arch Index cuando el usuario arrastra un punto en la GUI.
    # nuevo_sup_rgb y nuevo_inf_rgb vienen en el sistema de coordenadas de la imagen completa.
    
    # 1. Recuperamos los offsets (mínimos) para transformar de RGB a Máscara
    # Tu estructura guarda 'x_min' e 'y_min' que es la esquina donde empieza la máscara
    x_min = resultado_ai["arch_index"]["x_min"]
    y_min = resultado_ai["arch_index"]["y_min"]
    fila_corte = resultado_ai["arch_index"].get("fila_corte", 0)  # <-- NUEVO

    # 2. CONVERSIÓN CORRECTA A MÁSCARA: Restar y_min y fila_corte en Y
    y_offset_total = y_min + fila_corte
    nuevo_sup_mascara = (nuevo_sup_rgb[0] - x_min, nuevo_sup_rgb[1] - y_offset_total)
    nuevo_inf_mascara = (nuevo_inf_rgb[0] - x_min, nuevo_inf_rgb[1] - y_offset_total)

    # 3. CÁLCULO MATEMÁTICO (Usando los puntos convertidos a la máscara)
    linea1_mascara, linea2_mascara = calcular_divisiones_arch_index(nuevo_sup_mascara, nuevo_inf_mascara)
    areaA, areaB, areaC = calcular_areas_arch_index(mascara, nuevo_sup_mascara, nuevo_inf_mascara)

    # Evitamos división por cero por seguridad si la máscara está vacía
    if (areaA + areaB + areaC) == 0:
        nuevo_ai = 0.0
    else:
        nuevo_ai = areaB / (areaA + areaB + areaC)

# 4. CONVERSIÓN DE VUELTA A RGB: Sumar el offset total (y_min + fila_corte)
    (l1_inicio_x, l1_inicio_y), (l1_fin_x, l1_fin_y) = linea1_mascara
    (l2_inicio_x, l2_inicio_y), (l2_fin_x, l2_fin_y) = linea2_mascara

    linea1_rgb = ((int(l1_inicio_x + x_min), int(l1_inicio_y + y_offset_total)),
                  (int(l1_fin_x + x_min), int(l1_fin_y + y_offset_total)))
    linea2_rgb = ((int(l2_inicio_x + x_min), int(l2_inicio_y + y_offset_total)),
                  (int(l2_fin_x + x_min), int(l2_fin_y + y_offset_total)))
    
    # 4. GUARDAR PARA EL DIBUJO (Guardamos los puntos RGB para OpenCV)
    resultado_ai["arch_index"]["punto_superior"] = nuevo_sup_rgb
    resultado_ai["arch_index"]["punto_inferior"] = nuevo_inf_rgb
    
    # También guardamos las versiones en máscara por si otra función las requiere
    resultado_ai["arch_index"]["punto_superior_mascara"] = nuevo_sup_mascara
    resultado_ai["arch_index"]["punto_inferior_mascara"] = nuevo_inf_mascara

    # Guardamos las nuevas líneas y el valor calculado
    resultado_ai["arch_index"]["linea1"] = linea1_rgb
    resultado_ai["arch_index"]["linea2"] = linea2_rgb
    resultado_ai["arch_index"]["valor"] = nuevo_ai
    
    # 5. PRINTS DE DIAGNÓSTICO (Para controlar en la consola)
    print("\n--- DIAGNÓSTICO EN ARRASTRE ---")
    print("Sup (RGB):", nuevo_sup_rgb, " -> Sup (Máscara):", nuevo_sup_mascara)
    print("Inf (RGB):", nuevo_inf_rgb, " -> Inf (Máscara):", nuevo_inf_mascara)
    print(f"Áreas calculadas -> A: {areaA} | B: {areaB} | C: {areaC}")
    print(f"Nuevo Arch Index calculado: {nuevo_ai:.3f}")
    print("--------------------------------\n")

    return nuevo_ai

# Recalcula el Chippaux-Smirak Index (CSI) de forma dinámica cuando el usuario
# desplaza los puntos de control mediales en la GUI, actualizando el eje biomecánico,
# el barrido de perfiles transversales y los anchos críticos del antepié y mediopié.
def recalcular_chippaux_smirak(resultado_csi, mascara, nuevo_ante_rgb, nuevo_retro_rgb):
    
    csi_dict = resultado_csi["csi"]

    # 1. SINCRONIZACIÓN EXACTA DE OFFSETS (Reemplaza cv2.boundingRect)
    x_offset = csi_dict["x_min"]
    y_offset = csi_dict["y_min"] + csi_dict.get("fila_corte", 0)

    # 2. CONVERSIÓN A COORDENADAS LOCALES DE LA MÁSCARA RECORTADA
    ante_m = (
        max(0, int(nuevo_ante_rgb[0] - x_offset)),
        max(0, int(nuevo_ante_rgb[1] - y_offset)),
    )
    retro_m = (
        max(0, int(nuevo_retro_rgb[0] - x_offset)),
        max(0, int(nuevo_retro_rgb[1] - y_offset)),
    )

    # 3. CÁLCULO MATEMÁTICO (En tus funciones del backend)
    secciones = generar_secciones_mediales(ante_m, retro_m)
    perfil, extremos = calcular_perfil_anchuras(mascara, secciones)

    ancho_ante, ancho_medio, _, idx_ante, idx_medio, _ = obtener_anchos_indices(
        perfil, secciones
    )

    # Cálculo del índice Chippaux-Smirak
    nuevo_csi_val = (ancho_medio / ancho_ante) * 100 if ancho_ante > 0 else 0.0

    seg_ante_m = extremos[idx_ante]
    seg_medio_m = extremos[idx_medio]

    # 4. EJE MEDIAL VISUAL (Extensión vectorial estética en la GUI)
    # Evita que la recta fugue hacia el techo al cambiar la inclinación
    dx = nuevo_ante_rgb[0] - nuevo_retro_rgb[0]
    dy = nuevo_ante_rgb[1] - nuevo_retro_rgb[1]

    eje_sup_g = (int(nuevo_ante_rgb[0] + dx * 0.25), int(nuevo_ante_rgb[1] + dy * 0.25))
    eje_inf_g = (
        int(nuevo_retro_rgb[0] - dx * 0.25),
        int(nuevo_retro_rgb[1] - dy * 0.25),
    )

    # =========================================================================
    # 5. CONVERSIÓN DE VUELTA A RGB GLOBAL (Utilizando los offsets corregidos)
    # =========================================================================
    ancho_antepie_rgb = (
        (int(seg_ante_m[0][0] + x_offset), int(seg_ante_m[0][1] + y_offset)),
        (int(seg_ante_m[1][0] + x_offset), int(seg_ante_m[1][1] + y_offset)),
    ) if seg_ante_m is not None else None

    ancho_mediopie_rgb = (
        (int(seg_medio_m[0][0] + x_offset), int(seg_medio_m[0][1] + y_offset)),
        (int(seg_medio_m[1][0] + x_offset), int(seg_medio_m[1][1] + y_offset)),
    ) if seg_medio_m is not None else None
    # =========================================================================

    # 6. ACTUALIZAR EL DICCIONARIO VIVO PARA OPENCV
    csi_dict["punto_medial_antepie"] = nuevo_ante_rgb
    csi_dict["punto_medial_retropie"] = nuevo_retro_rgb
    csi_dict["eje_medial"] = (eje_sup_g, eje_inf_g)
    csi_dict["ancho_antepie"] = ancho_antepie_rgb
    csi_dict["ancho_mediopie"] = ancho_mediopie_rgb
    csi_dict["valor"] = nuevo_csi_val

    return nuevo_csi_val

# Recalcula el índice de Staheli de forma dinámica cuando el usuario desplaza
# los puntos de control mediales en la GUI, actualizando el eje biomecánico,
# el barrido de perfiles transversales y los anchos críticos del mediopié y retropié.
def recalcular_staheli(resultado_si, mascara, nuevo_ante_rgb, nuevo_retro_rgb):
    
    si_dict = resultado_si["staheli"]

    # 1. SINCRONIZACIÓN EXACTA DE OFFSETS (Reemplaza cv2.boundingRect)
    x_offset = si_dict["x_min"]
    y_offset = si_dict["y_min"] + si_dict.get("fila_corte", 0)

    # 2. CONVERSIÓN A COORDENADAS LOCALES (Protección contra índices negativos)
    ante_m = (
        max(0, int(nuevo_ante_rgb[0] - x_offset)),
        max(0, int(nuevo_ante_rgb[1] - y_offset)),
    )
    retro_m = (
        max(0, int(nuevo_retro_rgb[0] - x_offset)),
        max(0, int(nuevo_retro_rgb[1] - y_offset)),
    )

    # 3. CÁLCULO MATEMÁTICO ANATÓMICO (Lógica interna del backend)
    secciones = generar_secciones_mediales(ante_m, retro_m)
    perfil, extremos = calcular_perfil_anchuras(mascara, secciones)

    # Obtenemos anchos (descartamos antepié que no se usa en Staheli)
    _, ancho_medio, ancho_retro, _, idx_medio, idx_retro = (
        obtener_anchos_indices(perfil, secciones)
    )

    # Cálculo del índice de Staheli: mediopié / retropié
    nuevo_si_val = ancho_medio / ancho_retro if ancho_retro > 0 else 0.0

    seg_medio_m = extremos[idx_medio]
    seg_retro_m = extremos[idx_retro]

    # 4. EJE MEDIAL VISUAL (Extensión vectorial estética en la GUI)
    # Controla que la recta mantenga una proporción fija del 25% más allá de los puntos,
    # impidiendo deformaciones o fugas analíticas verticales hacia los bordes.
    dx = nuevo_ante_rgb[0] - nuevo_retro_rgb[0]
    dy = nuevo_ante_rgb[1] - nuevo_retro_rgb[1]

    eje_sup_g = (int(nuevo_ante_rgb[0] + dx * 0.25), int(nuevo_ante_rgb[1] + dy * 0.25))
    eje_inf_g = (
        int(nuevo_retro_rgb[0] - dx * 0.25),
        int(nuevo_retro_rgb[1] - dy * 0.25),
    )

    # =========================================================================
    # 5. CONVERSIÓN DE VUELTA A RGB GLOBAL (Sumando offsets corregidos)
    # =========================================================================
    ancho_mediopie_rgb = (
        (int(seg_medio_m[0][0] + x_offset), int(seg_medio_m[0][1] + y_offset)),
        (int(seg_medio_m[1][0] + x_offset), int(seg_medio_m[1][1] + y_offset)),
    ) if seg_medio_m is not None else None

    ancho_retropie_rgb = (
        (int(seg_retro_m[0][0] + x_offset), int(seg_retro_m[0][1] + y_offset)),
        (int(seg_retro_m[1][0] + x_offset), int(seg_retro_m[1][1] + y_offset)),
    ) if seg_retro_m is not None else None
    # =========================================================================

    # 6. GUARDAR CAMBIOS PARA EL DIBUJO DE OPENCV
    si_dict["punto_medial_antepie"] = nuevo_ante_rgb
    si_dict["punto_medial_retropie"] = nuevo_retro_rgb
    si_dict["eje_medial"] = (eje_sup_g, eje_inf_g)
    si_dict["ancho_mediopie"] = ancho_mediopie_rgb
    si_dict["ancho_retropie"] = ancho_retropie_rgb
    si_dict["valor"] = nuevo_si_val

    return nuevo_si_val

# Carga una imagen, la divide verticalmente para aislar cada
# pie y ejecuta de forma paralela todo el flujo biomecánico de preprocesamiento,
# orientación por PCA, segmentación de dedos y cálculo geométrico de los tres índices clínicos.
def procesar_imagen(nombre_archivo):
    imagen_pil = Image.open(nombre_archivo)  #Abro el archivo, usando la funcion open de la libreria Pillow
    imagen_rgb_original = imagen_pil.convert("RGB") #La imagen que abri la convierto en RGB, en caso que la original tuviera transparencia RGBA
    imagen_rgb = np.array(imagen_rgb_original) #Convierto lo que era un objeto imagen en una matriz de numeros usando la libreria NumPy
    
    alto, ancho = imagen_rgb.shape[:2]  #Me guardo solo el alto y el ancho de la matriz
    mitad = ancho // 2  #Hago la division entera del ancho
    
    # print(f"NOMBRE DE LA IMAGEN: {nombre_archivo}")
    
    imagen_izquierda = imagen_rgb[:, :mitad] #La imagen original la divido en dos y me quedo con la parte izquierda
    imagen_derecha = imagen_rgb[:, mitad:] #y la parte derecha, para trabajar con cada pie por separado

    # FIGURA 1: Imagen original
    # plt.figure(figsize=(8,8))
    # plt.imshow(imagen_rgb)
    # plt.title(f"Imagen original: {nombre_archivo}")
    # plt.axis('off')
    # plt.show()    
    
    
    huella_izquierda, umbral_izq = procesar_huella(imagen_izquierda)
    huella_derecha, umbral_der = procesar_huella(imagen_derecha)
            
    # dibujar_eje_pca(huella_izquierda,"PCA - Pie izquierdo")
    # dibujar_eje_pca(huella_derecha,"PCA - Pie derecho")
    
    angulo_izq = calcular_angulo_pca(huella_izquierda)
    angulo_der = calcular_angulo_pca(huella_derecha)
    # print(f"Ángulo pie izquierdo: {angulo_izq:.2f}")
    # print(f"Ángulo pie derecho: {angulo_der:.2f}")
    
    huella_izquierda_rotada = rotar_huella(huella_izquierda,angulo_izq)
    huella_derecha_rotada = rotar_huella(huella_derecha,angulo_der)
    
    huella_izquierda_recortada, x_min_izq, y_min_izq = recortar_huella(huella_izquierda_rotada)
    huella_derecha_recortada, x_min_der, y_min_der = recortar_huella(huella_derecha_rotada)    
    
    # fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    # # Huella izquierda
    # axes[0].imshow(huella_izquierda_recortada, cmap='gray')
    # axes[0].set_title("Pie izquierdo recortado")
    # axes[0].axis('off')
    # # Huella derecha
    # axes[1].imshow(huella_derecha_recortada, cmap='gray')
    # axes[1].set_title("Pie derecho recortado")
    # axes[1].axis('off')
    # plt.tight_layout()
    # plt.show()
    
    perfil_izquierdo = calcular_perfil_anchura(huella_izquierda_recortada)
    perfil_derecho = calcular_perfil_anchura(huella_derecha_recortada)
    perfil_izquierdo_suave = suavizar_perfil(perfil_izquierdo)
    perfil_derecho_suave = suavizar_perfil(perfil_derecho)
    
    corte_izquierdo = detectar_corte_dedos(perfil_izquierdo_suave)
    corte_derecho = detectar_corte_dedos(perfil_derecho_suave)
    fila_corte_izq = corte_izquierdo
    fila_corte_der = corte_derecho
    
    huella_izquierda_sin_dedos = eliminar_dedos(huella_izquierda_recortada,corte_izquierdo)
    huella_derecha_sin_dedos = eliminar_dedos(huella_derecha_recortada,corte_derecho)

    # fig, axes = plt.subplots(1, 2, figsize=(10, 10))    
    # axes[0].imshow(huella_izquierda_sin_dedos,cmap='gray')    
    # axes[0].set_title("Pie izquierdo sin dedos")    
    # axes[0].axis('off')    
    # axes[1].imshow(huella_derecha_sin_dedos,cmap='gray')
    # axes[1].set_title("Pie derecho sin dedos")
    # axes[1].axis('off')    
    # plt.tight_layout()
    # plt.show()

    contornos_izq = obtener_contorno(huella_izquierda_sin_dedos)
    contornos_der = obtener_contorno(huella_derecha_sin_dedos)            
    
    sup_izq, inf_izq = puntos_extremos_eje_longitudinal(huella_izquierda_sin_dedos)
    sup_der, inf_der = puntos_extremos_eje_longitudinal(huella_derecha_sin_dedos)

    linea1_izq, linea2_izq = calcular_divisiones_arch_index(sup_izq,inf_izq)
    linea1_der, linea2_der = calcular_divisiones_arch_index(sup_der,inf_der)    
    
    area_A_izq, area_B_izq, area_C_izq = calcular_areas_arch_index(huella_izquierda_sin_dedos,sup_izq,inf_izq)
    area_A_der, area_B_der, area_C_der = calcular_areas_arch_index(huella_derecha_sin_dedos,sup_der,inf_der)
    
    arch_index_izq = (area_B_izq /(area_A_izq + area_B_izq + area_C_izq))
    arch_index_der = (area_B_der /(area_A_der + area_B_der + area_C_der))    
    # print("\n--- ARCH INDEX ---")
    # print(f"Pie izquierdo: {arch_index_izq:.2f}")
    # print(f"Pie derecho: {arch_index_der:.2f}\n")    

    imagen_ai_izq = colorear_regiones_arch_index(huella_izquierda_sin_dedos,sup_izq,inf_izq)
    imagen_ai_der = colorear_regiones_arch_index(huella_derecha_sin_dedos,sup_der,inf_der)    
    
    
    medial_ante_izq, medial_retro_izq = puntos_eje_medial(huella_izquierda_sin_dedos,sup_izq,inf_izq,"izquierdo")
    medial_ante_der, medial_retro_der = puntos_eje_medial(huella_derecha_sin_dedos,sup_der,inf_der,"derecho")
    
    eje_sup_izq, eje_inf_izq = extender_eje_medial(medial_ante_izq,medial_retro_izq,sup_izq,inf_izq)
    eje_sup_der, eje_inf_der = extender_eje_medial(medial_ante_der,medial_retro_der,sup_der,inf_der)
    
    # secciones_izq = generar_secciones_mediales(medial_ante_izq,medial_retro_izq)
    # secciones_der = generar_secciones_mediales(medial_ante_der,medial_retro_der)
    secciones_izq = generar_secciones_mediales(eje_sup_izq,eje_inf_izq)
    secciones_der = generar_secciones_mediales(eje_sup_der,eje_inf_der)    
    # secciones_izq = generar_secciones_mediales(eje_sup_izq,eje_inf_izq,lado="izquierdo")
    # secciones_der = generar_secciones_mediales(eje_sup_der,eje_inf_der,lado="derecho")            
        
    perfil_izq, extremos_izq = calcular_perfil_anchuras(huella_izquierda_sin_dedos,secciones_izq)
    perfil_der, extremos_der = calcular_perfil_anchuras(huella_derecha_sin_dedos,secciones_der)
        
    ante_izq,medio_izq,retro_izq,idx_ante_izq,idx_medio_izq,idx_retro_izq = obtener_anchos_indices(perfil_izq,secciones_izq)
    ante_der,medio_der,retro_der,idx_ante_der,idx_medio_der,idx_retro_der = obtener_anchos_indices(perfil_der,secciones_der)
    
    segmento_ante_izq = extremos_izq[idx_ante_izq]
    segmento_medio_izq = extremos_izq[idx_medio_izq]
    segmento_retro_izq = extremos_izq[idx_retro_izq]
    segmento_ante_der = extremos_der[idx_ante_der]
    segmento_medio_der = extremos_der[idx_medio_der]
    segmento_retro_der = extremos_der[idx_retro_der]

    csi_izq = (medio_izq /ante_izq) * 100
    csi_der = (medio_der /ante_der) * 100

    si_izq = (medio_izq /retro_izq)
    si_der = (medio_der /retro_der)
    
    # print("\n--- CHIPPAUX-SMIRAK ---")
    # print(f"Izquierdo: {csi_izq:.2f}")
    # print(f"Derecho: {csi_der:.2f}")

    # print("\n--- STAHELI ---")
    # print(f"Izquierdo: {si_izq:.2f}")
    # print(f"Derecho: {si_der:.2f}")    
    
    linea_ante_izq = obtener_linea_seccion(secciones_izq,idx_ante_izq)    
    linea_medio_izq = obtener_linea_seccion(secciones_izq,idx_medio_izq)
    linea_retro_izq = obtener_linea_seccion(secciones_izq,idx_retro_izq)    
    linea_ante_der = obtener_linea_seccion(secciones_der,idx_ante_der)
    linea_medio_der = obtener_linea_seccion(secciones_der,idx_medio_der)    
    linea_retro_der = obtener_linea_seccion(secciones_der,idx_retro_der)


    imagen_izquierda_rgb_rotada = rotar_imagen_rgb(imagen_izquierda,angulo_izq)
    imagen_derecha_rgb_rotada = rotar_imagen_rgb(imagen_derecha,angulo_der)    

    # punto_superior_izq_rgb = transformar_punto_a_rgb(sup_izq,x_min_izq,y_min_izq,fila_corte_izq)    
    # punto_inferior_izq_rgb = transformar_punto_a_rgb(inf_izq,x_min_izq,y_min_izq,fila_corte_izq)
    # punto_superior_der_rgb = transformar_punto_a_rgb(sup_der,x_min_der,y_min_der,fila_corte_der)
    # punto_inferior_der_rgb = transformar_punto_a_rgb(inf_der,x_min_der,y_min_der,fila_corte_der)    
    
    
    contornos_izq_rgb = [
        transformar_contorno_a_rgb(c, x_min_izq, y_min_izq, fila_corte_izq)
        for c in contornos_izq
    ]    
    contornos_der_rgb = [
        transformar_contorno_a_rgb(c, x_min_der, y_min_der, fila_corte_der)
        for c in contornos_der
    ]


    sup_izq_rgb = transformar_punto_a_rgb(sup_izq,x_min_izq,y_min_izq,fila_corte_izq)
    inf_izq_rgb = transformar_punto_a_rgb(inf_izq,x_min_izq,y_min_izq,fila_corte_izq)
    linea1_izq_rgb = transformar_segmento_a_rgb(linea1_izq,x_min_izq,y_min_izq,fila_corte_izq)
    linea2_izq_rgb = transformar_segmento_a_rgb(linea2_izq,x_min_izq,y_min_izq,fila_corte_izq)
    
    sup_der_rgb = transformar_punto_a_rgb(sup_der,x_min_der,y_min_der,fila_corte_der)
    inf_der_rgb = transformar_punto_a_rgb(inf_der,x_min_der,y_min_der,fila_corte_der)
    linea1_der_rgb = transformar_segmento_a_rgb(linea1_der,x_min_der,y_min_der,fila_corte_der)
    linea2_der_rgb = transformar_segmento_a_rgb(linea2_der,x_min_der,y_min_der,fila_corte_der)
    
    resultado_izq_AI = {
        "arch_index": {
            "valor": arch_index_izq,
            "punto_superior": sup_izq_rgb,
            "punto_inferior": inf_izq_rgb,
            "punto_superior_mascara": sup_izq,
            "punto_inferior_mascara": inf_izq,
            "linea1": linea1_izq_rgb,
            "linea2": linea2_izq_rgb,
            "contornos": contornos_izq_rgb,
            "x_min": x_min_izq,
            "y_min": y_min_izq,
            "fila_corte": fila_corte_izq
        }
    }
    resultado_der_AI = {
        "arch_index": {
            "valor": arch_index_der,
            "punto_superior": sup_der_rgb,
            "punto_inferior": inf_der_rgb,
            "punto_superior_mascara": sup_der,
            "punto_inferior_mascara": inf_der,
            "linea1": linea1_der_rgb,
            "linea2": linea2_der_rgb,
            "contornos": contornos_der_rgb,
            "x_min": x_min_der,
            "y_min": y_min_der,
            "fila_corte": fila_corte_der
        }   
    }

    
    
    medial_ante_izq_rgb = transformar_punto_a_rgb(medial_ante_izq,x_min_izq,y_min_izq,fila_corte_izq)
    medial_retro_izq_rgb = transformar_punto_a_rgb(medial_retro_izq,x_min_izq,y_min_izq,fila_corte_izq)
    eje_medial_izq_rgb = transformar_segmento_a_rgb((eje_sup_izq, eje_inf_izq),x_min_izq,y_min_izq,fila_corte_izq)
    segmento_ante_izq_rgb = transformar_segmento_a_rgb(segmento_ante_izq,x_min_izq,y_min_izq,fila_corte_izq)
    if segmento_medio_izq is not None:
        segmento_medio_izq_rgb = transformar_segmento_a_rgb(segmento_medio_izq,x_min_izq,y_min_izq,fila_corte_izq)    
    else:    
        segmento_medio_izq_rgb = None
    segmento_retro_izq_rgb = transformar_segmento_a_rgb(segmento_retro_izq,x_min_izq,y_min_izq,fila_corte_izq)
    
    medial_ante_der_rgb = transformar_punto_a_rgb(medial_ante_der,x_min_der,y_min_der,fila_corte_der)
    medial_retro_der_rgb = transformar_punto_a_rgb(medial_retro_der,x_min_der,y_min_der,fila_corte_der)
    eje_medial_der_rgb = transformar_segmento_a_rgb((eje_sup_der, eje_inf_der),x_min_der,y_min_der,fila_corte_der)
    segmento_ante_der_rgb = transformar_segmento_a_rgb(segmento_ante_der,x_min_der,y_min_der,fila_corte_der)
    if segmento_medio_der is not None:
        segmento_medio_der_rgb = transformar_segmento_a_rgb(segmento_medio_der,x_min_der,y_min_der,fila_corte_der)
    else:    
        segmento_medio_der_rgb = None
    segmento_retro_der_rgb = transformar_segmento_a_rgb(segmento_retro_der,x_min_der,y_min_der,fila_corte_der)

    resultado_izq_CSI={
        "csi":{
            "valor": csi_izq,        
            "punto_medial_antepie": medial_ante_izq_rgb,
            "punto_medial_retropie": medial_retro_izq_rgb,        
            "eje_medial": eje_medial_izq_rgb,        
            "ancho_antepie": segmento_ante_izq_rgb,
            "ancho_mediopie": segmento_medio_izq_rgb,
            "contornos": contornos_izq_rgb,
            "x_min": x_min_izq,
            "y_min": y_min_izq,
            "fila_corte": fila_corte_izq,
            "y_sup_mascara": sup_izq[1], # Guardamos la altura Y superior de la huella
            "y_inf_mascara": inf_izq[1]  # Guardamos la altura Y inferior de la huella
        }
    }    
    resultado_der_CSI={
        "csi":{
            "valor": csi_der,
            "punto_medial_antepie": medial_ante_der_rgb,
            "punto_medial_retropie": medial_retro_der_rgb,        
            "eje_medial": eje_medial_der_rgb,        
            "ancho_antepie": segmento_ante_der_rgb,
            "ancho_mediopie": segmento_medio_der_rgb,        
            "contornos": contornos_der_rgb,
            "x_min": x_min_der,
            "y_min": y_min_der,
            "fila_corte": fila_corte_der,
            "y_sup_mascara": sup_der[1],
            "y_inf_mascara": inf_der[1]
        }
    }



    resultado_izq_SI= {
        "staheli":{
            "valor": si_izq,    
            "punto_medial_antepie": medial_ante_izq_rgb,
            "punto_medial_retropie": medial_retro_izq_rgb,    
            "eje_medial": eje_medial_izq_rgb,    
            "ancho_retropie": segmento_retro_izq_rgb,
            "ancho_mediopie": segmento_medio_izq_rgb,    
            "contornos": contornos_izq_rgb,
            "x_min": x_min_izq,
            "y_min": y_min_izq,
            "fila_corte": fila_corte_izq,
            "y_sup_mascara": sup_izq[1],
            "y_inf_mascara": inf_izq[1]
        }
    }    
    resultado_der_SI= {
        "staheli":{
            "valor": si_der,    
            "punto_medial_antepie": medial_ante_der_rgb,
            "punto_medial_retropie": medial_retro_der_rgb,    
            "eje_medial": eje_medial_der_rgb,    
            "ancho_retropie": segmento_retro_der_rgb,
            "ancho_mediopie": segmento_medio_der_rgb,    
            "contornos": contornos_der_rgb,
            "x_min": x_min_der,
            "y_min": y_min_der,
            "fila_corte": fila_corte_der,
            "y_sup_mascara": sup_der[1],
            "y_inf_mascara": inf_der[1]
        }
    }

    
    return {
        "imagen_izq": imagen_izquierda_rgb_rotada,
        "imagen_der": imagen_derecha_rgb_rotada,
        "angulo_izq": angulo_izq,
        "angulo_der": angulo_der,
        "mascara_izq": huella_izquierda_sin_dedos,
        "mascara_der": huella_derecha_sin_dedos,
        "AI_izq": resultado_izq_AI,
        "AI_der": resultado_der_AI,
        "CSI_izq": resultado_izq_CSI,
        "CSI_der": resultado_der_CSI,
        "SI_izq": resultado_izq_SI,
        "SI_der": resultado_der_SI
    }


# ==================================================
# PROGRAMA PRINCIPAL
# ==================================================

if __name__ == "__main__":
    for i in range(30,31):
        nombre_archivo = f"8va_{i}.png"
        huella = procesar_imagen(nombre_archivo)
        
        # 2. Consumís los datos usando resultados["clave"]
        # --- Gráficos ARCH INDEX ---
        # graficar_arch_index(huella["imagen_izq"], huella["AI_izq"])
        # graficar_arch_index(huella["imagen_der"], huella["AI_der"])
                
        # # --- Gráficos CSI ---
        # graficar_csi(huella["imagen_izq"], huella["CSI_izq"])
        # graficar_csi(huella["imagen_der"], huella["CSI_der"])
                
        # # --- Gráficos STAHELI ---
        # graficar_staheli(huella["imagen_izq"], huella["SI_izq"])
        # graficar_staheli(huella["imagen_der"], huella["SI_der"])
