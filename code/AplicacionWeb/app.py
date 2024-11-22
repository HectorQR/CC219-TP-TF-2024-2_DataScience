import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuración de la página
st.set_page_config(
    page_title="Recomendador de peliculas",
    page_icon="🎬",
    layout="wide"
)

# API de TMDB
TMDB_API_KEY = "6793a79efb88deefc30af7b5f746ccb4"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Extraer datos imgen de TMDB
def get_movie_details(title):

    # Buscar
    search_url = f"{TMDB_BASE_URL}/search/movie"
    response = requests.get(search_url, params={
        'api_key': TMDB_API_KEY,
        'query': title,
        'language': 'es-ES'
    })
    
    if response.status_code == 200:
        results = response.json().get('results', [])
        if results:
            movie = results[0]  #primer resultado
            return {
                'poster_path': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie['poster_path'] else None,
                'overview': movie['overview']
            }
    return {
        'poster_path': None,
        'overview': "No hay descripción disponible."
    }

# Cargar estilos CSS desde archivo
def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Cargar y preparar data
@st.cache_data
def cargar_datos(ruta_csv):
    try:
        df = pd.read_csv(ruta_csv)
        modelo = df[['title', 'overview_clean', 'genres', 'director']].copy()
        modelo['genres'] = modelo['genres'].fillna('').apply(
            lambda x: ' '.join(x.replace(',', ' ').replace('-', '').lower().split()))
        modelo['director'] = modelo['director'].fillna('').apply(
            lambda x: ' '.join(x.replace(',', ' ').replace('-', '').lower().split()))
        return modelo
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

# Generar recomendaciones
def get_recomendacion_m1(titulo, modelo, tfidf_matriz):
    try:
        indices = pd.Series(modelo.index, index=modelo['title']).drop_duplicates()
        if titulo not in indices:
            return ["Película no encontrada"]
        
        idx = indices[titulo]
        if modelo.duplicated(['title']).any():
            idx = modelo[modelo['title'] == titulo].index[0]
        
        simil = sorted(
            enumerate(cosine_similarity(tfidf_matriz[idx], tfidf_matriz).flatten()),
            key=lambda x: x[1],
            reverse=True
        )[1:6]
        
        # Obtener imagen de TMDB
        recomendaciones = []
        for i in simil:
            titulo_pelicula = modelo.iloc[i[0]]['title']
            detalles_tmdb = get_movie_details(titulo_pelicula)
            recomendaciones.append({
                'title': titulo_pelicula,
                'overview': detalles_tmdb['overview'],
                'poster_path': detalles_tmdb['poster_path']
            })
        return recomendaciones
    
    except Exception as e:
        st.error(f"Error al generar recomendaciones: {e}")
        return []

def main():
    load_css('styles.css')
    
    # Título principal
    st.markdown("""
        <h1 class='stTitle'>
            🎬 Recomendador de peliculas
        </h1>
    """, unsafe_allow_html=True)

    # Descripción
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <p style='font-size: 1.2rem; color: #4A5568;'>
               Ingresa el titulo de una pelicual para recomendarte opciones basadas en tus gustos cinematográficos.
        </div>
    """, unsafe_allow_html=True)

    # Cargar archivo
    ruta_csv = "movies_dataset_clean.csv"
    modelo1 = cargar_datos(ruta_csv)

    if modelo1 is not None:
        # Crear TF-IDF y calcular matriz
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_matriz_1 = tfidf.fit_transform(
            modelo1['overview_clean'] + ' ' + modelo1['genres'] + ' ' + modelo1['director'])

        # Entrada de usuario
        peliculas_disponibles = modelo1['title'].tolist()
        peli = st.selectbox(
            "🔍 Busca una película",
            options=peliculas_disponibles,
            index=peliculas_disponibles.index("The Dark Knight Rises") if "The Dark Knight Rises" in peliculas_disponibles else 0
        )

        # Botón de búsqueda
        if st.button("🎯 Buscar similares", key="recomendar"):
            with st.spinner('Buscando las mejores recomendaciones para ti...'):
                recomendaciones = get_recomendacion_m1(peli, modelo1, tfidf_matriz_1)
                
                if recomendaciones:
                    st.markdown("<h3 class='recommendation-header'>📽️ Películas recomendadas</h3>", unsafe_allow_html=True)
                    
                    for i, pelicula in enumerate(recomendaciones, 1):
                        poster_url = pelicula['poster_path'] if pelicula['poster_path'] else "/api/placeholder/150/225"
                        # Tarjetas para cada peli
                        st.markdown(f"""
                            <div class='movie-card'>
                                <div class='movie-poster'>
                                    <img src="{poster_url}" 
                                         alt="Póster de {pelicula['title']}"
                                         style="width: 100%; height: 100%; object-fit: cover;">
                                </div>
                                <div class='movie-content'>
                                    <h4 class='movie-title'>
                                        {i}. {pelicula['title']}
                                    </h4>
                                    <p class='movie-description'>
                                        {pelicula['overview']}
                                    </p>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No se encontraron recomendaciones para esta película.")
    else:
        st.error("No se pudo cargar el dataset")
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()