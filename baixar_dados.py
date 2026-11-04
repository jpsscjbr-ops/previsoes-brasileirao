import pandas as pd

print("Conectando ao banco de dados público do Brasileirão...")

# Link direto para o famoso repositório 'Brasileirao_Dataset' (sempre online)
url = "https://raw.githubusercontent.com/adaoduque/Brasileirao_Dataset/master/campeonato-brasileiro-full.csv"

# O Pandas baixa a tabela inteira sem bloqueios
df_brasil = pd.read_csv(url)

# Salva o arquivo no seu Mac
df_brasil.to_csv("jogos_brasileirao.csv", index=False)

print("✅ Sucesso Absoluto! O arquivo 'jogos_brasileirao.csv' foi criado na sua pasta.")
print(f"Foram salvos {len(df_brasil)} jogos históricos!")