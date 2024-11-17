# CC219-TP-TF-2024-2_DataScience
# Sistema de Recomendación De Peliculas

## Objetivo del Trabajo
Este proyecto desarrolla un sistema de recomendación basado en contenido para plataformas de streaming. Su propósito es ayudar a los usuarios a encontrar contenido relevante analizando características intrínsecas de las películas, como género, duración, y popularidad, para superar la "paradoja de la elección".

## Autores
- **Josafat Larios Mellado** – U20201B124  
- **Diego Armando Flores Carrizales** – U201922066  
- **Hector Jesus Quintana Robatti** – U20201B280  

## Descripción del Dataset
El dataset utilizado fue obtenido de [Kaggle](https://www.kaggle.com/datasets/utkarshx27/movies-dataset?select=movie_dataset.csv) y contiene más de 45,000 registros relacionados con información cinematográfica. Incluye atributos como:
- `budget`: Presupuesto en dólares.
- `genres`: Géneros asociados.
- `original_language`: Idioma original.
- `popularity`: Popularidad de las películas.
- `vote_average` y `vote_count`: Calificación promedio y cantidad de votos.
- Entre otros.

El archivo procesado final, `movies_dataset_clean.csv`, contiene datos limpios y listos para análisis, eliminando columnas con datos nulos o irrelevantes para el modelo.

## Conclusiones
1. El modelo de similitud coseno basado en características intrínsecas es eficaz para recomendaciones precisas, especialmente cuando se priorizan géneros y directores similares.
2. Las técnicas como Regresión Lineal Múltiple y Random Forest muestran el potencial predictivo del dataset, especialmente para calificaciones y géneros populares.
3. Este sistema tiene aplicaciones comerciales significativas, mejorando la experiencia del usuario y la fidelización en plataformas de streaming.

## Licencia
Este proyecto está licenciado bajo la [MIT License](LICENSE). Puedes utilizar, modificar y distribuir el contenido respetando los términos establecidos.

---

## Archivos Relevantes
- **Código del Modelo:** Disponible en [Google Colab](https://colab.research.google.com/drive/1NIh7r9RfgDPBzX40jhvjUahv1UZARPLa?usp=sharing)
