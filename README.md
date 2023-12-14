# proyecto-mineria-de-datos
Proyecto para el curso Minería de Datos (CC5205) de la FCFM de la Universidad de Chile

## Integrantes

* [Carla Guzman](https://github.com/CarlaGuzmanR)
* [Paola Silva](https://github.com/Paito249)
* [Pablo Skewes](https://github.com/pabloskewes/)
* [Isaias Venegas](https://github.com/IsaiasVenegas)
* [Cynthia Vega](https://github.com/Cynthia-Vega)


## Descripción del proyecto

El proyecto se enmarca en el curso Minería de Datos de la FCFM de la Universidad de Chile. El objetivo por un lado es predecir un posible incremento en la inscripción de mujeres en el área STEM y por el otro agrupar a las instituciones de educación superior según relaciones no triviales entre sus características (puntajes de corte, infraestructura, cantidad de docentes, etc.), todo a partir de los Índices Educación Superior (IES) del Ministerio de Educación de Chile. Para esto, se utilizarán técnicas de aprendizaje supervisado (regresión lineal) y no supervisado (k-means), además de técnicas de visualización de datos.

## Dataset

Para el desarrollo del proyecto se utilizará el dataset de los Índices Educación Superior (IES) del Ministerio de Educación de Chile. Este dataset contiene información de las instituciones de educación superior del país, como si se encuentran acreditadas, tipo de admisión, sedes, entre otros. El dataset se puede encontrar en el siguiente [link](https://cned.cl/institucional/bases-de-datos/) bajo el nombre de "Base INDICES Matrícula 2005-2023".

Durante el desarrollo del proyecto se vió la necesidad de agregar información por institución. El dataset adicional se puede encontrar en el mismo sitio bajo el nombre de "INDICES Institucional 2005-2022".

## Instalación

Para poder correr el proyecto es necesario tener instalado Python 3 superior. Se recomienda usar un entorno virtual para instalar las dependencias del proyecto. Para esto, se puede usar `virtualenv` o `conda`. Para instalar las dependencias, por ejemplo se puede hacer lo siguiente:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Ejecución
Una vez hecho la instalación de librerias se recomienda usar el editor de código Visual Studio junto con la extension de Microsoft "Jupyter". Al ejecutar el primer bloque de código se deberá seleccionar el entorno virtual creado.
