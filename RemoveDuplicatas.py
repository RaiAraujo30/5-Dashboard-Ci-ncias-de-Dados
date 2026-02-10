# Ler o arquivo de cidades
with open('cidades.txt', 'r', encoding='utf-8') as file:
    cities = [line.strip() for line in file.readlines()]

# Remover duplicatas usando set e manter a ordem
unique_cities = sorted(set(cities), key=cities.index)

# Escrever o arquivo de cidades distintas
with open('cidades_distintas.txt', 'w', encoding='utf-8') as output_file:
    for city in unique_cities:
        output_file.write(f"{city}\n")

print("O arquivo de cidades distintas foi gerado com sucesso!")
