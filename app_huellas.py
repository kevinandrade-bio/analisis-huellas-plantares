import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
# Importo el procesador y las funciones de dibujo por OpenCV
from backend_huellas import (
    procesar_imagen,
    dibujar_retroalimentacion_ai,
    dibujar_retroalimentacion_csi,
    dibujar_retroalimentacion_si,
    recalcular_arch_index,
    recalcular_chippaux_smirak,
    recalcular_staheli
)
import math
import cv2
import os  # <-- AGREGAR ESTE IMPORT

ruta_imagen = None
imagen_izq_tk = None
imagen_der_tk = None
imagen_original_tk = None
resultados_actuales = None
punto_seleccionado = None
escala_izq_x = 1
escala_izq_y = 1
escala_der_x = 1
escala_der_y = 1
offset_izq_x = 0
offset_izq_y = 0
offset_der_x = 0
offset_der_y = 0

# Abre un cuadro de diálogo para seleccionar una imagen (.png/.jpg), la escala a un 
# tamaño máximo de 700x500 manteniendo la proporción, la muestra en la interfaz y 
# reinicia el estado de los contenedores visuales para ocultar resultados previos.
def cargar_imagen():
    global ruta_imagen, imagen_original_tk, resultados_actuales, punto_seleccionado

    ruta = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=[("Imágenes", "*.png")])
    if ruta == "":
        return

    ruta_imagen = ruta
    
    # --- NUEVO: Extraer solo el nombre y actualizar la interfaz ---
    nombre_archivo = os.path.basename(ruta)
    archivo_cargado_var.set(f"{nombre_archivo}")
    # -------------------------------------------------------------
    
    # Reiniciamos variables de cálculo por si venimos de procesar una imagen anterior
    resultados_actuales = None
    punto_seleccionado = None

    imagen = Image.open(ruta)
    imagen.thumbnail((700, 500))  # Escalo la imagen al tamaño que quiero que se vea
    imagen_original_tk = ImageTk.PhotoImage(imagen)  # Transforma la imagen PIL al formato de Tkinter

    label_original.config(image=imagen_original_tk)
    label_original.image = imagen_original_tk

    # 1. DESMONTAMOS TODOS LOS CONTENEDORES (Clave para resetear el orden vertical en Tkinter)
    label_inicial.pack_forget()   # Oculta el cartel inicial ("Cargue una imagen...")
    frame_imagenes.pack_forget()  # Oculta las imágenes procesadas si venimos de un análisis anterior
    frame_tarjetas.pack_forget()  # Desmonta las tarjetas para volver a darles prioridad arriba
    frame_original.pack_forget()  # <-- ¡ESTA ES LA CLAVE! Desmonta la foto original para que no le gane el lugar arriba a las tarjetas

    # 2. EMPAQUETAMOS EN ORDEN ESTRICTO DE ARRIBA HACIA ABAJO
    frame_tarjetas.pack(fill="x", pady=(0, 5))  # 1° Las tarjetas van arriba de todo
    frame_original.pack(expand=True, pady=10)   # 2° La foto original va justo debajo de las tarjetas
    
    # 3. REINICIAMOS LOS TEXTOS DE LAS TARJETAS
    val_tarjeta_izq.set("--")
    diag_tarjeta_izq.set("Sin procesar")
    val_tarjeta_der.set("--")
    diag_tarjeta_der.set("Sin procesar")

# Convierte la imagen PIL en una matriz de NumPy para detectar de forma eficiente 
# los píxeles negros puros (RGB = 0, 0, 0), los reemplaza por el color morado 
# especificado y devuelve el resultado transformado nuevamente en una imagen PIL.
def cambiar_fondo_negro(img_pil, color_morado=(126, 74, 134)):
    
    data = np.array(img_pil)# Convierte la imagen a una matriz de numpy para que sea rápido
    fondo_negro = (data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0) # Encuentra donde R, G y B son todos 0 (negro puro)
    data[fondo_negro] = color_morado # Asigna el color morado donde encontro negro
    return Image.fromarray(data) #Transforma la matriz a imagen

# Controla el flujo principal de procesamiento: valida la carga de la imagen, ejecuta el backend 
# biomecánico y, según el método clínico seleccionado (Arch Index, CSI o Staheli), dibuja las 
# líneas métricas correspondientes. Luego, cambia el fondo negro por morado, escala los resultados 
# centrándolos en los Canvas y calcula las escalas y offsets necesarios para el mapeo de coordenadas
def procesar():
    #Comprueba que exista un archivo cargado antes de correr el algoritmo
    global ruta_imagen, imagen_izq_tk, imagen_der_tk
    if ruta_imagen is None: #Si apreta procesar antes de cargar imagen aparece ese mensaje
        label_resultado.config(text="No hay imagen cargada") 
        return

    #Envía la ruta de la imagen a las funciones de procesamiento matemático y guarda el 
    #diccionario/struct de resultados en una variable global. Luego, extrae qué 
    #método clínico quiere ver el usuario y aísla las matrices de imagen base de cada pie.
    global resultados_actuales
    resultados_actuales= procesar_imagen(ruta_imagen) #Llama a la funcion que hice en el archivo de procesamiento
    resultados = resultados_actuales
    metodo_actual = metodo.get() #Lee que opcion de los metodos tiene apretada el usuario en el panel de control
    print("Método:", metodo_actual)    
    img_izq_matriz = resultados["imagen_izq"] # Guarda del struct de resultados las imágenes limpias rotadas
    img_der_matriz = resultados["imagen_der"]

    # Determina que dibujar dependiendo de la opcion que selecciono el usuario en la GUI
    if metodo_actual == "arch_index":
        #Busca los cálculos matemáticos correspondientes en el struct
        val_izq = resultados["AI_izq"]["arch_index"]["valor"]
        val_der = resultados["AI_der"]["arch_index"]["valor"]
        texto = f"Arch Index - Izquierdo: {val_izq:.3f} | Derecho: {val_der:.3f}"
        #Llamo las funciones encargadas de dibujar las lineas de cada pie
        img_izq_final = dibujar_retroalimentacion_ai(img_izq_matriz, resultados["AI_izq"])
        img_der_final = dibujar_retroalimentacion_ai(img_der_matriz, resultados["AI_der"])
        
    elif metodo_actual == "csi":
        #Busca los cálculos matemáticos correspondientes en el struct
        val_izq = resultados["CSI_izq"]["csi"]["valor"]
        val_der = resultados["CSI_der"]["csi"]["valor"]
        texto = f"Chippaux-Smirak - Izquierdo: {val_izq:.2f}% | Derecho: {val_der:.2f}%"
        #Llamo las funciones encargadas de dibujar las lineas de cada pie
        img_izq_final = dibujar_retroalimentacion_csi(img_izq_matriz, resultados["CSI_izq"])
        img_der_final = dibujar_retroalimentacion_csi(img_der_matriz, resultados["CSI_der"])
        
    elif metodo_actual == "staheli":
        #Busca los cálculos matemáticos correspondientes en el struct
        val_izq = resultados["SI_izq"]["staheli"]["valor"]
        val_der = resultados["SI_der"]["staheli"]["valor"]
        texto = f"Staheli - Izquierdo: {val_izq:.2f} | Derecho: {val_der:.2f}"   
        #Llamo las funciones encargadas de dibujar las lineas de cada pie
        img_izq_final = dibujar_retroalimentacion_si(img_izq_matriz, resultados["SI_izq"])
        img_der_final = dibujar_retroalimentacion_si(img_der_matriz, resultados["SI_der"])

    # label_resultado.config(text=texto)
    actualizar_tarjetas_resultados()
    
    #Guarda el tamaño real de las imágenes procesadas (para la escala más adelante). 
    alto_original_izq, ancho_original_izq = img_izq_final.shape[:2]
    alto_original_der, ancho_original_der = img_der_final.shape[:2]    

    #Convertir las matrices pintadas por OpenCV a imágenes PIL
    img_izq_pil = Image.fromarray(img_izq_final).convert("RGB")
    img_der_pil = Image.fromarray(img_der_final).convert("RGB")
    
    #Como se grafican las imagenes rotadas, queda un fondo negro entonces lo pinto morado
    img_izq_pil = cambiar_fondo_negro(img_izq_pil)
    img_der_pil = cambiar_fondo_negro(img_der_pil)
    
    # Redimensionar para que se visualicen correctamente en la pantalla
    img_izq_pil.thumbnail((448, 616))
    img_der_pil.thumbnail((448, 616))
    print("Tamaño mostrado izquierda:", img_izq_pil.size)
    print("Tamaño mostrado derecha:", img_der_pil.size)
    
    ancho_mostrado_izq, alto_mostrado_izq = img_izq_pil.size
    ancho_mostrado_der, alto_mostrado_der = img_der_pil.size
    
    #Las imágenes se achican manteniendo la proporción, rara vez miden exactamente 480x660. 
    #Este cálculo saca los "márgenes vacíos" que quedan alrededor de la imagen dentro del Canvas
    #Si hago click en el lienzo, necesito restar los offset para saber a qué punto real del pie le estás pegando
    global offset_izq_x
    global offset_izq_y
    global offset_der_x
    global offset_der_y
    
    offset_izq_x = (448 - ancho_mostrado_izq) / 2
    offset_izq_y = (616 - alto_mostrado_izq) / 2
    
    offset_der_x = (448 - ancho_mostrado_der) / 2
    offset_der_y = (616 - alto_mostrado_der) / 2
    
    print("Offset izquierda:", offset_izq_x, offset_izq_y)
    print("Offset derecha:", offset_der_x, offset_der_y)

    #Calcula cuántas veces más grande es la imagen original que la que estoy viendo en pantalla. 
    #Fundamental para mapear coordenadas de clicks de la GUI hacia la matriz real de procesamiento
    global escala_izq_x
    global escala_izq_y
    global escala_der_x
    global escala_der_y
    
    escala_izq_x = ancho_original_izq / ancho_mostrado_izq
    escala_izq_y = alto_original_izq / alto_mostrado_izq
    
    escala_der_x = ancho_original_der / ancho_mostrado_der
    escala_der_y = alto_original_der / alto_mostrado_der
    
    print("Escala izquierda:", escala_izq_x, escala_izq_y)
    print("Escala derecha:", escala_der_x, escala_der_y)

    #Transforma las imágenes finales al formato compatible con Tkinter
    imagen_izq_tk = ImageTk.PhotoImage(img_izq_pil) 
    imagen_der_tk = ImageTk.PhotoImage(img_der_pil)

    #Limpia cualquier imagen previa que hubiese para que no se encimen con los resultados nuevos
    canvas_izq.delete("all") 
    canvas_der.delete("all")
    
    #Estampa la imagen en el centro del lienzo
    canvas_izq.create_image(224, 308, anchor="center", image=imagen_izq_tk) 
    canvas_der.create_image(224, 308, anchor="center", image=imagen_der_tk)
    
    #Oculta la foto original que cargó al principio
    frame_original.pack_forget() 
    #Muestra el nuevo contenedor que ahora tiene los dos lienzos con los pies izquierdo y derecho analizados
    frame_imagenes.pack(pady=0) 

# Gestiona el evento de clic en el Canvas del pie izquierdo: mapea las coordenadas del clic 
# a la imagen real y busca si el usuario seleccionó un punto de control del método clínico 
# activo (AI, CSI o SI). Si encuentra un punto cercano, guarda sus datos en 
# la variable global y fuerza el redibujado para resaltar la selección.
def click_canvas_izq(event):

    global resultados_actuales, punto_seleccionado
    
    # Si se hace clic antes de que existan resultados calculados, cancela la acción
    if resultados_actuales is None:
        return
    
    # Obtiene el método clínico seleccionado actualmente en la GUI
    metodo_actual = metodo.get()

    if metodo_actual == "arch_index":
        # Convierte los píxeles del clic en el Canvas a las coordenadas reales de la imagen
        x_real, y_real = canvas_a_imagen_izq(event.x,event.y)        
        print(f"Canvas=({event.x},{event.y}) " f"Imagen=({x_real:.1f},{y_real:.1f})")        
        # Verifica si el clic está lo suficientemente cerca de algún punto clave del metodo
        nombre = buscar_punto_ai(x_real,y_real,resultados_actuales["AI_izq"])
        
        if nombre is not None:
            print(f"Punto seleccionado: {nombre}")
            #Estructura la información del punto seleccionado para el arrastre posterior
            punto_seleccionado = {
                "metodo": "arch_index",
                "pie": "izq",
                "nombre": nombre,
                "coord": resultados_actuales["AI_izq"]["arch_index"][nombre]
            }
            #Actualiza los lienzos para pintar el feedback visual (círculo rojo) sobre el punto
            redibujar_imagenes()
        else:
            print("No se seleccionó ningún punto")
    
    elif metodo_actual == "csi":
        # Convierte las coordenadas del clic para el análisis de Chippaux-Smirak
        x_real, y_real = canvas_a_imagen_izq(event.x, event.y)        
        # Busca si el clic coincide con algún punto de control de CSI
        nombre = buscar_punto_csi(x_real, y_real, resultados_actuales["CSI_izq"])
        if nombre is not None:
            # Almacena los datos del punto de control de CSI encontrado y fuerza el redibujado
            punto_seleccionado = {
                "metodo": "csi", 
                "pie": "izq", 
                "nombre": nombre, 
                "coord": resultados_actuales["CSI_izq"]["csi"][nombre]
            }
            redibujar_imagenes()
    elif metodo_actual == "staheli":
        # Convierte las coordenadas del clic para el análisis del índice de Staheli
        x_real, y_real = canvas_a_imagen_izq(event.x, event.y)        
        # Busca si el clic coincide con algún punto de control de Staheli
        nombre = buscar_punto_si(x_real, y_real, resultados_actuales["SI_izq"])
        if nombre is not None:
            # Almacena los datos del punto de control de Staheli encontrado y fuerza el redibujado
            punto_seleccionado = {
                "metodo": "staheli", 
                "pie": "izq", 
                "nombre": nombre, 
                "coord": resultados_actuales["SI_izq"]["staheli"][nombre]
            }
            redibujar_imagenes()

def click_canvas_der(event):

    global resultados_actuales, punto_seleccionado

    if resultados_actuales is None:
        return

    metodo_actual = metodo.get()

    if metodo_actual == "arch_index":
        # Conversión usando la escala y offset del pie derecho
        x_real, y_real = canvas_a_imagen_der(event.x, event.y)        
        print(f"Canvas Der=({event.x},{event.y}) Imagen Der=({x_real:.1f},{y_real:.1f})")        
        
        # Buscamos el punto en el sub-diccionario del pie derecho
        nombre = buscar_punto_ai(x_real, y_real, resultados_actuales["AI_der"])
        
        if nombre is not None:
            print(f"Punto derecho seleccionado: {nombre}")
            coord_punto = resultados_actuales["AI_der"]["arch_index"][nombre]
            punto_seleccionado = {
                "metodo": "arch_index",
                "pie": "der",
                "nombre": nombre,
                "coord": coord_punto
            }
            redibujar_imagenes()
        else:
            print("No se seleccionó ningún punto en el pie derecho")
            
    elif metodo_actual == "csi":  # <-- NUEVO BLOQUE
        x_real, y_real = canvas_a_imagen_der(event.x, event.y)        
        nombre = buscar_punto_csi(x_real, y_real, resultados_actuales["CSI_der"])
        if nombre is not None:
            punto_seleccionado = {"metodo": "csi", "pie": "der", "nombre": nombre, "coord": resultados_actuales["CSI_der"]["csi"][nombre]}
            redibujar_imagenes()
    elif metodo_actual == "staheli":
        x_real, y_real = canvas_a_imagen_der(event.x, event.y)        
        nombre = buscar_punto_si(x_real, y_real, resultados_actuales["SI_der"])
        if nombre is not None:
            punto_seleccionado = {"metodo": "staheli", "pie": "der", "nombre": nombre, "coord": resultados_actuales["SI_der"]["staheli"][nombre]}
            redibujar_imagenes()
            
# Transforma las coordenadas de un punto capturado en el Canvas del pie izquierdo a las 
# coordenadas reales de la imagen original, restando los márgenes de centrado (offsets) 
# y multiplicando por los factores de escala correspondientes en ambos ejes
def canvas_a_imagen_izq(x_canvas, y_canvas):

    # Resta los márgenes vacíos (offsets) para alinear el origen (0,0) del clic con el inicio real de la imagen
    x_canvas = x_canvas - offset_izq_x
    y_canvas = y_canvas - offset_izq_y
    # Multiplica por los factores de escala para convertir los píxeles de pantalla a píxeles de la matriz original
    x_real = x_canvas * escala_izq_x
    y_real = y_canvas * escala_izq_y
    # Devuelve el par de coordenadas mapeadas a la resolución real del backend
    return (x_real, y_real)

def canvas_a_imagen_der(x_canvas, y_canvas):

    x_canvas = x_canvas - offset_der_x
    y_canvas = y_canvas - offset_der_y

    x_real = x_canvas * escala_der_x
    y_real = y_canvas * escala_der_y

    return (x_real, y_real)

# Calcula la distancia euclidiana en línea recta entre dos puntos (p1 y p2)
def distancia(p1, p2):

    return math.sqrt((p1[0] - p2[0])**2 +(p1[1] - p2[1])**2)

# Evalúa si las coordenadas reales de un clic se encuentran dentro de un radio de tolerancia de 
# 20 píxeles respecto a los puntos clave del método Arch Index (punto superior o inferior) 
# y, de ser así, devuelve el nombre del punto seleccionado para habilitar su edición.
def buscar_punto_ai(click_x, click_y, resultado_ai):

    # Accede al sub-diccionario que contiene los cálculos e índices del Arch Index
    ai = resultado_ai["arch_index"]
    # Agrupa los puntos de control anatómicos que el usuario tiene permitido seleccionar
    puntos = {
        "punto_superior": ai["punto_superior"],
        "punto_inferior": ai["punto_inferior"]
    }
    radio = 20
    # Recorre cada punto geométrico de la lista para chequear las distancias
    for nombre, punto in puntos.items():
        # Si la distancia  entre el clic y el punto real es menor al radio de tolerancia
        if distancia((click_x, click_y),punto) < radio:
            # Devuelve la clave del punto atrapado (ej. "punto_superior")
            return nombre
    # Si el clic fue afuera de la zona devuelve None
    return None

def buscar_punto_csi(click_x, click_y, resultado_csi):
    csi = resultado_csi["csi"]
    puntos = {
        "punto_medial_antepie": csi["punto_medial_antepie"],
        "punto_medial_retropie": csi["punto_medial_retropie"]
    }
    radio = 20
    for nombre, punto in puntos.items():
        if distancia((click_x, click_y), punto) < radio:
            return nombre
    return None

def buscar_punto_si(click_x, click_y, resultado_si):
    si = resultado_si["staheli"]
    puntos = {
        "punto_medial_antepie": si["punto_medial_antepie"],
        "punto_medial_retropie": si["punto_medial_retropie"]
    }
    radio = 20
    for nombre, punto in puntos.items():
        if distancia((click_x, click_y), punto) < radio:
            return nombre
    return None
    
# Actualiza la visualización de los Canvas: genera copias limpias de las matrices originales, 
# pinta las líneas métricas del método activo y el punto rojo de control si hay una selección, 
# procesa el color del fondo y escala los resultados para estamparlos en la GUI.
def redibujar_imagenes():
    
    global resultados_actuales

    if resultados_actuales is None:
        return

    metodo_actual = metodo.get()
    
    # Trabaja sobre copias para no arruinar las matrices limpias originales con múltiples dibujos sucesivos
    img_izq_matriz = resultados_actuales["imagen_izq"].copy()
    img_der_matriz = resultados_actuales["imagen_der"].copy()

    #Invoca las funciones de dibujo específicas para el método activo en la interfaz
    if metodo_actual == "arch_index":
        img_izq_final = dibujar_retroalimentacion_ai(img_izq_matriz,resultados_actuales["AI_izq"])
        img_der_final = dibujar_retroalimentacion_ai(img_der_matriz,resultados_actuales["AI_der"])

    elif metodo_actual == "csi":
        img_izq_final = dibujar_retroalimentacion_csi(img_izq_matriz,resultados_actuales["CSI_izq"])
        img_der_final = dibujar_retroalimentacion_csi(img_der_matriz,resultados_actuales["CSI_der"])

    else:
        img_izq_final = dibujar_retroalimentacion_si(img_izq_matriz,resultados_actuales["SI_izq"])
        img_der_final = dibujar_retroalimentacion_si(img_der_matriz,resultados_actuales["SI_der"])

    #Si el usuario está interactuando con un punto, dibuja un círculo rojo de 7px para dar feedback visual
    if punto_seleccionado is not None:
        x = int(punto_seleccionado["coord"][0])
        y = int(punto_seleccionado["coord"][1])
        
        # Elegimos sobre qué matriz pintar el círculo azul de control
        if punto_seleccionado["pie"] == "izq":
            cv2.circle(img_izq_final, (x, y), 7, (255, 0, 0), -1)
        else:
            cv2.circle(img_der_final, (x, y), 7, (255, 0, 0), -1)

    # Convierte los resultados finales de matrices de OpenCV a objetos de imagen PIL
    img_izq_pil = Image.fromarray(img_izq_final)
    img_der_pil = Image.fromarray(img_der_final)
    # Reemplaza los fondos negros por el color morado estético de la aplicación
    img_izq_pil = cambiar_fondo_negro(img_izq_pil)
    img_der_pil = cambiar_fondo_negro(img_der_pil)
    # Redimensiona las imágenes para que encajen en las dimensiones de los Canvas de la pantalla
    img_izq_pil.thumbnail((448,616))
    img_der_pil.thumbnail((448,616))

    global imagen_izq_tk
    global imagen_der_tk
    
    # Transforma las imágenes finales a objetos PhotoImage para que sean compatibles con Tkinter
    imagen_izq_tk = ImageTk.PhotoImage(img_izq_pil)
    imagen_der_tk = ImageTk.PhotoImage(img_der_pil)
    
    # Borra el cuadro anterior de los lienzos gráficos para refrescar el renderizado
    canvas_izq.delete("all")
    canvas_der.delete("all")

    # En ambas funciones, cambiar (200, 275) por (240, 330)
    canvas_izq.create_image(224, 308, anchor="center", image=imagen_izq_tk) 
    canvas_der.create_image(224, 308, anchor="center", image=imagen_der_tk)
    
# Sigue el movimiento del mouse durante el arrastre: calcula la posición real del cursor en la imagen, 
# actualiza las coordenadas del punto de control en el backend en tiempo real y fuerza el redibujado 
# continuo para proveer una animación fluida de las líneas guía.
def arrastrar_punto(event):
    global punto_seleccionado
    global resultados_actuales
    
    # Si no hay ningún punto de control seleccionado previamente en el clic, ignora el arrastre
    if punto_seleccionado is None:
        return

    pie = punto_seleccionado["pie"]
    metodo_actual = punto_seleccionado["metodo"]
    nombre = punto_seleccionado["nombre"]

    #Convierte la posición del cursor a píxeles reales de la imagen
    if pie == "izq":
        x_real, y_real = canvas_a_imagen_izq(event.x, event.y)
        if metodo_actual == "arch_index": resultado_metodo = resultados_actuales["AI_izq"]
        elif metodo_actual == "csi": resultado_metodo = resultados_actuales["CSI_izq"]
        else: resultado_metodo = resultados_actuales["SI_izq"]
    else:
        x_real, y_real = canvas_a_imagen_der(event.x, event.y)
        if metodo_actual == "arch_index": resultado_metodo = resultados_actuales["AI_der"]
        elif metodo_actual == "csi": resultado_metodo = resultados_actuales["CSI_der"]
        else: resultado_metodo = resultados_actuales["SI_der"]

    #Traduce el nombre del método clínico al formato de clave que utiliza el diccionario del backend
    clave_interna = "arch_index" if metodo_actual == "arch_index" else ("csi" if metodo_actual == "csi" else "staheli")
    
    #Actualizamos la posición en tiempo real para estirar la línea verde guía
    resultado_metodo[clave_interna][nombre] = (x_real, y_real)
    punto_seleccionado["coord"] = (x_real, y_real)

    redibujar_imagenes()
    
# Concluye el arrastre al soltar el clic del mouse: ejecuta los algoritmos geométricos pesados del backend 
# sobre las máscaras segmentadas usando las coordenadas finales, actualiza las etiquetas con los nuevos 
# índices numéricos y libera la variable de selección borrando el círculo de control.
def soltar_punto(event):
    global punto_seleccionado
    global resultados_actuales
    
    # Cancela la ejecución si no se estaba arrastrando ningún punto de control válido
    if punto_seleccionado is None:
        return

    pie = punto_seleccionado["pie"]
    metodo_actual = punto_seleccionado["metodo"]

    # --- PROCESAMIENTO GEOMÉTRICO PARA ARCH INDEX ---
    if metodo_actual == "arch_index":
        if pie == "izq":
            resultado_ai = resultados_actuales["AI_izq"]
            mascara = resultados_actuales.get("mascara_izq", resultado_ai["arch_index"].get("mascara", None))
        else:
            resultado_ai = resultados_actuales["AI_der"]
            mascara = resultados_actuales.get("mascara_der", resultado_ai["arch_index"].get("mascara", None))

        if mascara is not None:
            # Tomamos los puntos en la posición final donde los dejó el usuario
            sup_rgb = resultado_ai["arch_index"]["punto_superior"]
            inf_rgb = resultado_ai["arch_index"]["punto_inferior"]
            
            # Llamamos al backend para procesar la máscara y actualizar las líneas definitivas
            recalcular_arch_index(resultado_ai, mascara, sup_rgb, inf_rgb)
            
            # Actualizamos el panel de texto superior con los nuevos índices calculados
            val_izq = resultados_actuales["AI_izq"]["arch_index"]["valor"]
            val_der = resultados_actuales["AI_der"]["arch_index"]["valor"]
            # label_resultado.config(text=f"Arch Index - Izquierdo: {val_izq:.3f} | Derecho: {val_der:.3f}")
        else:
            print(f"Error: No se pudo procesar la máscara al soltar el pie {pie}")
    
    # --- PROCESAMIENTO GEOMÉTRICO PARA CHIPPAUX-SMIRAK ---
    elif metodo_actual == "csi":  
        if pie == "izq":
            resultado_csi = resultados_actuales["CSI_izq"]
            mascara = resultados_actuales.get("mascara_izq", None)
        else:
            resultado_csi = resultados_actuales["CSI_der"]
            mascara = resultados_actuales.get("mascara_der", None)

        if mascara is not None:
            # Captura las posiciones finales de los puntos mediales de antepié y retropié escogidos
            ante_rgb = resultado_csi["csi"]["punto_medial_antepie"]
            retro_rgb = resultado_csi["csi"]["punto_medial_retropie"]
            # Llamamos al backend para procesar la máscara y actualizar las líneas definitivas
            recalcular_chippaux_smirak(resultado_csi, mascara, ante_rgb, retro_rgb)
            # Actualizamos el panel de texto superior con los nuevos índices calculados
            val_izq = resultados_actuales["CSI_izq"]["csi"]["valor"]
            val_der = resultados_actuales["CSI_der"]["csi"]["valor"]
            # label_resultado.config(text=f"Chippaux-Smirak - Izquierdo: {val_izq:.2f}% | Derecho: {val_der:.2f}%")   
    
    # --- PROCESAMIENTO GEOMÉTRICO PARA STAHELI ---
    elif metodo_actual == "staheli": 
        if pie == "izq":
            resultado_si = resultados_actuales["SI_izq"]
            mascara = resultados_actuales.get("mascara_izq", None)
        else:
            resultado_si = resultados_actuales["SI_der"]
            mascara = resultados_actuales.get("mascara_der", None)

        if mascara is not None:
            ante_rgb = resultado_si["staheli"]["punto_medial_antepie"]
            retro_rgb = resultado_si["staheli"]["punto_medial_retropie"]
            recalcular_staheli(resultado_si, mascara, ante_rgb, retro_rgb)
            val_izq = resultados_actuales["SI_izq"]["staheli"]["valor"]
            val_der = resultados_actuales["SI_der"]["staheli"]["valor"]
            # label_resultado.config(text=f"Staheli - Izquierdo: {val_izq:.2f} | Derecho: {val_der:.2f}")
    
    # Forzar el recalculo de los diagnósticos al soltar el arrastre de un punto
    actualizar_tarjetas_resultados()
    
    # Liberamos el control del punto seleccionado
    punto_seleccionado = None
    
    # Redibujamos por última vez para quitar el círculo rojo de control y posicionar las líneas en su lugar recalculado
    redibujar_imagenes()
    
# Devuelve el tipo de pie según las reglas de Arch Index
def obtener_diagnostico_ai(valor):
    if valor < 0.21:
        return "Pie Cavo"
    elif 0.21 <= valor < 0.25:
        return "Pie Normal"
    else:
        return "Pie Plano"
    
def obtener_diagnostico_csi(valor):
    # Recibe porcentaje (ej: 35.5%)
    if valor < 25.0:
        return "Pie Cavo"
    elif 25.0 <= valor < 45.0:
        return "Pie Normal"
    else:
        return "Pie Plano"
    
def obtener_diagnostico_si(valor):
    # Ajustado a escala decimal (0.20 en vez de 20%) para que coincida con el backend
    if valor < 0.5:
        return "Pie Cavo"
    elif 0.5 <= valor < 0.7:
        return "Pie Normal"
    else:
        return "Pie Plano"

# Actualiza dinámicamente la guía en el Panel de Control
def actualizar_guia_visual(*args):
    # Primero ocultamos todas para evitar solapamientos
    frame_guia_ai.pack_forget()
    frame_guia_csi.pack_forget()
    frame_guia_si.pack_forget()
    
    metodo_actual = metodo.get()
    
    if metodo_actual == "arch_index":
        frame_guia_ai.pack(pady=15, fill="x", padx=15)
    elif metodo_actual == "csi":
        frame_guia_csi.pack(pady=15, fill="x", padx=15)
    elif metodo_actual == "staheli":
        frame_guia_si.pack(pady=15, fill="x", padx=15)

# Centraliza la actualización de títulos y diagnósticos
def actualizar_titulos_diagnosticos():
    
    texto_titulo_izq.set("Pie Izquierdo")
    texto_titulo_der.set("Pie Derecho")

# Actualiza los valores grandes y diagnósticos en las tarjetas visuales
def actualizar_tarjetas_resultados():
    if resultados_actuales is None:
        val_tarjeta_izq.set("--")
        diag_tarjeta_izq.set("Sin procesar")
        val_tarjeta_der.set("--")
        diag_tarjeta_der.set("Sin procesar")
        return

    metodo_actual = metodo.get()

    if metodo_actual == "arch_index":
        val_izq = resultados_actuales["AI_izq"]["arch_index"]["valor"]
        val_der = resultados_actuales["AI_der"]["arch_index"]["valor"]
        val_tarjeta_izq.set(f"{val_izq:.3f}")
        val_tarjeta_der.set(f"{val_der:.3f}")
        diag_tarjeta_izq.set(obtener_diagnostico_ai(val_izq))
        diag_tarjeta_der.set(obtener_diagnostico_ai(val_der))

    elif metodo_actual == "csi":
        val_izq = resultados_actuales["CSI_izq"]["csi"]["valor"]
        val_der = resultados_actuales["CSI_der"]["csi"]["valor"]
        val_tarjeta_izq.set(f"{val_izq:.2f}%")
        val_tarjeta_der.set(f"{val_der:.2f}%")
        diag_tarjeta_izq.set(obtener_diagnostico_csi(val_izq))
        diag_tarjeta_der.set(obtener_diagnostico_csi(val_der))

    elif metodo_actual == "staheli":
        val_izq = resultados_actuales["SI_izq"]["staheli"]["valor"]
        val_der = resultados_actuales["SI_der"]["staheli"]["valor"]
        val_tarjeta_izq.set(f"{val_izq:.2f}")
        val_tarjeta_der.set(f"{val_der:.2f}")
        diag_tarjeta_izq.set(obtener_diagnostico_si(val_izq))
        diag_tarjeta_der.set(obtener_diagnostico_si(val_der))


# ==================================================
# VENTANA PRINCIPAL DE TKINTER (REDISEÑADA)
# ==================================================
ventana = tk.Tk() #Crea la ventana principal del sistema
ventana.title("Herramienta para clasificacion de huella plantar")
ventana.geometry("1200x850") # Hacemos la ventana más ancha para las columnas
ventana.config(bg="#f5f5f5") # Fondo gris claro moderno

# Configuración de estilos modernos TTK
style = ttk.Style()
style.theme_use("clam")  # "clam" o "vista" permiten personalizar botones en Windows/Linux

# Estilo para botones principales
style.configure("Accion.TButton", font=("Arial", 10, "bold"), padding=8)
style.configure("Radio.TRadiobutton", font=("Arial", 9), background="#ffffff")


# --------------------------------------------------
# COLUMNA IZQUIERDA: Panel de Control (Menú Lateral)
# --------------------------------------------------
panel_izquierdo = tk.Frame(ventana, bg="#ffffff", width=280, relief="groove", borderwidth=1)
panel_izquierdo.pack(side="left", fill="y", padx=10, pady=10) #Ancla este bloque a la izquierda de la ventana principal y hace que se estire de arriba a abajo para ocupar todo el alto disponible
panel_izquierdo.pack_propagate(False) #Le dice a la interfaz "Este panel mide exactamente 280 píxeles de ancho, metas lo que metas no lo achiques ni deformes"

tk.Label(panel_izquierdo, text="PANEL DE CONTROL", font=("Arial", 12, "bold"), bg="#ffffff", fg="#333").pack(pady=15)

# Botón de carga con ttk
btn_cargar = ttk.Button(panel_izquierdo, text="Cargar Imagen", command=cargar_imagen,style="Accion.TButton",cursor="hand2")
btn_cargar.pack(pady=10, fill="x", padx=20)

# --- NUEVO: Variable y Label para mostrar el nombre del archivo ---
archivo_cargado_var = tk.StringVar(value="Ningún archivo seleccionado")

label_archivo = tk.Label(
    panel_izquierdo,
    textvariable=archivo_cargado_var,
    font=("Arial", 8, "italic"),
    fg="#6c757d",      # Tono gris discreto
    bg="#ffffff",
    wraplength=240,    # Evita que un nombre muy largo ensanche el panel
    justify="center"
)
# Si prefieres tu idea de ponerlo ABAJO DE TODO en la esquina inferior izquierda,
# cambia .pack(pady=(0, 10)) por .pack(side="bottom", pady=15)
label_archivo.pack(pady=(0, 10), padx=10)
# ------------------------------------------------------------------

# Contenedor de Métodos
frame_metodo = tk.LabelFrame(panel_izquierdo, text=" Método de Análisis ", font=("Arial", 10, "bold"), bg="#ffffff", fg="#555", padx=10, pady=10)
frame_metodo.pack(pady=15, fill="x", padx=15)

metodo = tk.StringVar(value="arch_index")
metodo.trace_add("write", actualizar_guia_visual)

ttk.Radiobutton(frame_metodo, text="Arch Index (AI)", variable=metodo, value="arch_index", style="Radio.TRadiobutton").pack(anchor="w", pady=4)
ttk.Radiobutton(frame_metodo, text="Chippaux-Smirak (CSI)", variable=metodo, value="csi", style="Radio.TRadiobutton").pack(anchor="w", pady=4)
ttk.Radiobutton(frame_metodo, text="Índice de Staheli (SI)", variable=metodo, value="staheli", style="Radio.TRadiobutton").pack(anchor="w", pady=4)

# Botón Procesar con ttk
btn_procesar = ttk.Button(panel_izquierdo, text="Procesar Huellas", command=procesar,style="Accion.TButton",cursor="hand2")
btn_procesar.pack(pady=20, fill="x", padx=20)

# --- CONTENEDOR GUÍA INFORMATIVA ARCH INDEX (AI) ---
frame_guia_ai = tk.LabelFrame(panel_izquierdo, text=" Guía de Rangos (AI) ", font=("Arial", 10, "bold"), bg="#ffffff", fg="#333333", padx=10, pady=10)

# Fila Pie Cavo
fila_ai_cavo = tk.Frame(frame_guia_ai, bg="#ffffff")
fila_ai_cavo.pack(fill="x", pady=2)
tk.Label(fila_ai_cavo, text="• Pie Cavo: ", font=("Arial", 9, "bold"), bg="#ffffff", fg="#007bff").pack(side="left")
tk.Label(fila_ai_cavo, text="AI < 0.210", font=("Arial", 9, "bold"), bg="#ffffff", fg="#333333").pack(side="left")

# Fila Pie Normal
fila_ai_norm = tk.Frame(frame_guia_ai, bg="#ffffff")
fila_ai_norm.pack(fill="x", pady=2)
tk.Label(fila_ai_norm, text="• Pie Normal: ", font=("Arial", 9, "bold"), bg="#ffffff", fg="#007bff").pack(side="left")
tk.Label(fila_ai_norm, text="0.210 ≤ AI ≤ 0.250", font=("Arial", 9, "bold"), bg="#ffffff", fg="#333333").pack(side="left")

# Fila Pie Plano
fila_ai_plan = tk.Frame(frame_guia_ai, bg="#ffffff")
fila_ai_plan.pack(fill="x", pady=2)
tk.Label(fila_ai_plan, text="• Pie Plano: ", font=("Arial", 9, "bold"), bg="#ffffff", fg="#007bff").pack(side="left")
tk.Label(fila_ai_plan, text="AI > 0.250", font=("Arial", 9, "bold"), bg="#ffffff", fg="#333333").pack(side="left")

# --- CONTENEDOR GUÍA INFORMATIVA CHIPPAUX-SMIRAK (CSI) ---
frame_guia_csi = tk.LabelFrame(panel_izquierdo, text=" Guía de Rangos (CSI) ", font=("Arial", 10, "bold"), bg="#ffffff", fg="#333333", padx=10, pady=10)

# Fila Pie Cavo
fila_csi_cavo = tk.Frame(frame_guia_csi, bg="#ffffff")
fila_csi_cavo.pack(fill="x", pady=2)
tk.Label(fila_csi_cavo, text="• Pie Cavo: ", font=("Arial", 9, "bold"), bg="#ffffff", fg="#007bff").pack(side="left")
tk.Label(fila_csi_cavo, text="CSI < 25%", font=("Arial", 9, "bold"), bg="#ffffff", fg="#333333").pack(side="left")

# Fila Pie Normal
fila_csi_norm = tk.Frame(frame_guia_csi, bg="#ffffff")
fila_csi_norm.pack(fill="x", pady=2)
tk.Label(fila_csi_norm, text="• Pie Normal: ", font=("Arial", 9, "bold"), bg="#ffffff", fg="#007bff").pack(side="left")
tk.Label(fila_csi_norm, text="25% ≤ CSI ≤ 45%", font=("Arial", 9, "bold"), bg="#ffffff", fg="#333333").pack(side="left")

# Fila Pie Plano
fila_csi_plan = tk.Frame(frame_guia_csi, bg="#ffffff")
fila_csi_plan.pack(fill="x", pady=2)
tk.Label(fila_csi_plan, text="• Pie Plano: ", font=("Arial", 9, "bold"), bg="#ffffff", fg="#007bff").pack(side="left")
tk.Label(fila_csi_plan, text="CSI > 45%", font=("Arial", 9, "bold"), bg="#ffffff", fg="#333333").pack(side="left")

# --- CONTENEDOR GUÍA INFORMATIVA INDEX STAHELI (SI) ---
frame_guia_si = tk.LabelFrame(panel_izquierdo, text=" Guía de Rangos (SI) ", font=("Arial", 10, "bold"), bg="#ffffff", fg="#333333", padx=10, pady=10)

# Fila Pie Cavo
fila_si_cavo = tk.Frame(frame_guia_si, bg="#ffffff")
fila_si_cavo.pack(fill="x", pady=2)
tk.Label(fila_si_cavo, text="• Pie Cavo: ", font=("Arial", 9, "bold"), bg="#ffffff", fg="#007bff").pack(side="left")
tk.Label(fila_si_cavo, text="SI < 0.50", font=("Arial", 9, "bold"), bg="#ffffff", fg="#333333").pack(side="left")

# Fila Pie Normal
fila_si_norm = tk.Frame(frame_guia_si, bg="#ffffff")
fila_si_norm.pack(fill="x", pady=2)
tk.Label(fila_si_norm, text="• Pie Normal: ", font=("Arial", 9, "bold"), bg="#ffffff", fg="#007bff").pack(side="left")
tk.Label(fila_si_norm, text="0.50 ≤ SI ≤ 0.70", font=("Arial", 9, "bold"), bg="#ffffff", fg="#333333").pack(side="left")

# Fila Pie Plano
fila_si_plan = tk.Frame(frame_guia_si, bg="#ffffff")
fila_si_plan.pack(fill="x", pady=2)
tk.Label(fila_si_plan, text="• Pie Plano: ", font=("Arial", 9, "bold"), bg="#ffffff", fg="#007bff").pack(side="left")
tk.Label(fila_si_plan, text="SI > 0.70", font=("Arial", 9, "bold"), bg="#ffffff", fg="#333333").pack(side="left")
# Forzamos que se dibuje inicialmente ya que arranca seteado en Arch Index por defecto
actualizar_guia_visual()


# --------------------------------------------------
# COLUMNA DERECHA: Panel de Visualización (Resultados)
# --------------------------------------------------
panel_derecho = tk.Frame(ventana, bg="#f5f5f5")
panel_derecho.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# 1. Título / Cartel inicial antes de cargar imagen
label_inicial = tk.Label(panel_derecho, text="Cargue una imagen para comenzar el análisis", font=("Arial", 14, "bold"), fg="#495057", bg="#e9ecef", pady=15, relief="flat")
label_inicial.pack(fill="x", pady=(0, 10))

# ==================================================
# CONTENEDOR DE TARJETAS DE RESULTADOS (PIE IZQ Y DER)
# ==================================================
frame_tarjetas = tk.Frame(panel_derecho, bg="#f5f5f5")

# Variables dinámicas para los textos de las tarjetas
val_tarjeta_izq = tk.StringVar(value="--")
diag_tarjeta_izq = tk.StringVar(value="Cargue una imagen")
val_tarjeta_der = tk.StringVar(value="--")
diag_tarjeta_der = tk.StringVar(value="Cargue una imagen")

# --- TARJETA PIE IZQUIERDO ---
card_izq = tk.Frame(frame_tarjetas, bg="#ffffff", bd=1, relief="solid", padx=15, pady=10)
card_izq.pack(side="left", fill="x", expand=True, padx=(0, 10))

tk.Label(card_izq, text="PIE IZQUIERDO", font=("Arial", 9, "bold"), fg="#6c757d", bg="#ffffff").pack(anchor="center")
tk.Label(card_izq, textvariable=val_tarjeta_izq, font=("Arial", 13, "bold"), fg="#495057", bg="#ffffff").pack(anchor="center", pady=2)
tk.Label(card_izq, textvariable=diag_tarjeta_izq, font=("Arial", 16, "bold"), fg="#343a40", bg="#ffffff").pack(anchor="center")

# --- TARJETA PIE DERECHO ---
card_der = tk.Frame(frame_tarjetas, bg="#ffffff", bd=1, relief="solid", padx=15, pady=10)
card_der.pack(side="left", fill="x", expand=True, padx=(10, 0))

tk.Label(card_der, text="PIE DERECHO", font=("Arial", 9, "bold"), fg="#6c757d", bg="#ffffff").pack(anchor="center")
tk.Label(card_der, textvariable=val_tarjeta_der, font=("Arial", 13, "bold"), fg="#495057", bg="#ffffff").pack(anchor="center", pady=2)
tk.Label(card_der, textvariable=diag_tarjeta_der, font=("Arial", 16, "bold"), fg="#343a40", bg="#ffffff").pack(anchor="center")

# Contenedor para la imagen original (Vista Inicial Completa)
frame_original = tk.Frame(panel_derecho, bg="#f5f5f5")
label_original = tk.Label(frame_original, bg="#f5f5f5")
label_original.pack(pady=20)
frame_original.pack(expand=True)

# Contenedor para la vista dividida (Izquierda y Derecha lado a lado)
frame_imagenes = tk.Frame(panel_derecho, bg="#f5f5f5")

# Creamos StringVars dinámicas para los textos superiores de las huellas
texto_titulo_izq = tk.StringVar(value="Análisis Pie Izquierdo")
texto_titulo_der = tk.StringVar(value="Análisis Pie Derecho")

frame_izq = tk.Frame(frame_imagenes, bg="#f5f5f5")
frame_izq.pack(side="left", padx=15, expand=True)
canvas_izq = tk.Canvas(frame_izq, width=448, height=616, bg="#f5f5f5", borderwidth=0, relief="solid")
canvas_izq.pack()
canvas_izq.bind("<Button-1>",click_canvas_izq)
canvas_izq.bind("<B1-Motion>", arrastrar_punto)
canvas_izq.bind("<ButtonRelease-1>", soltar_punto)

frame_der = tk.Frame(frame_imagenes, bg="#f5f5f5")
frame_der.pack(side="left", padx=15, expand=True)
canvas_der = tk.Canvas(frame_der, width=448, height=616, bg="#f5f5f5", borderwidth=0, relief="solid")
canvas_der.pack()
canvas_der.bind("<Button-1>",click_canvas_der)
canvas_der.bind("<B1-Motion>", arrastrar_punto) # <-- Asociamos también el derecho acá
canvas_der.bind("<ButtonRelease-1>", soltar_punto) # <-- Le asignamos el mismo soltar_punto

ventana.mainloop()