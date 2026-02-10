import geopy
from geopy.geocoders import Nominatim
import json

# Função para pegar as coordenadas de uma cidade
def get_coordinates(city, geolocator):
    try:
        location = geolocator.geocode(city)
        if location:
            return location.latitude, location.longitude
        else:
            return None
    except Exception as e:
        return None

# Ler o arquivo de cidades
with open('cidades_distintas.txt', 'r', encoding='utf-8') as file:
    cities = [line.strip() for line in file.readlines()]

# Inicializar o geolocalizador
geolocator = Nominatim(user_agent="geoapiExercises")

# Dicionário para armazenar as coordenadas
city_coordinates = {}

# Processar as cidades e obter coordenadas
total_cities = len(cities)
for index, city in enumerate(cities):
    print(f"Processando cidade {index + 1} de {total_cities}: {city}")  # Print do andamento
    coordinates = get_coordinates(city, geolocator)
    if coordinates:
        city_coordinates[city] = {
            "latitude": coordinates[0],
            "longitude": coordinates[1]
        }
    else:
        city_coordinates[city] = {
            "latitude": None,
            "longitude": None
        }

# Gerar o JSON de saída
with open('coordenadas.json', 'w', encoding='utf-8') as json_file:
    json.dump(city_coordinates, json_file, ensure_ascii=False, indent=4)

print("O arquivo de coordenadas foi gerado com sucesso!")
