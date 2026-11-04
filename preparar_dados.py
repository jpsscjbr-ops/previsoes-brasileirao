import pandas as pd

print("Lendo o cofre completo de jogos...")
df = pd.read_csv("jogos_brasileirao.csv")

# Garante que a coluna de data está no formato correto do Pandas
df['data'] = pd.to_datetime(df['data'])

print(f"Total de jogos antes do filtro: {len(df)}")

# 🎯 FILTRAGEM INTELIGENTE: Mantém apenas os jogos de 2024 em diante (últimas 3 temporadas)
df = df[df['data'].dt.year >= 2024].copy()

print(f"Total de jogos após filtrar de 2024 para cá: {len(df)}")

# Seleciona apenas as colunas essenciais
colunas_importantes = ['data', 'mandante', 'visitante', 'mandante_Placar', 'visitante_Placar']
df_limpo = df[colunas_importantes].dropna()

# Salva a versão otimizada
df_limpo.to_csv("jogos_limpos_para_ia.csv", index=False)
print("✅ Arquivo 'jogos_limpos_para_ia.csv' atualizado com foco apenas nas temporadas recentes!")