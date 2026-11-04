import pandas as pd

print("🌐 Conectando ao repositório público do Football-Data (100% gratuito e atualizado)...")

# Fonte pública e confiável que atualiza os CSVs de futebol diariamente
url = "https://www.football-data.co.uk/new/BRA.csv"
df = pd.read_csv(url)

# Ajusta as datas e filtra apenas de 2024 para frente
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date'])
df = df[df['Date'].dt.year >= 2024].copy()

# Padroniza os nomes das colunas para o nosso modelo de IA
df = df.rename(columns={
    'Date': 'data',
    'Home': 'mandante',
    'Away': 'visitante',
    'HG': 'mandante_Placar',
    'AG': 'visitante_Placar'
})

# Isola os jogos que já aconteceram e salva
df_jogos_reais = df[['data', 'mandante', 'visitante', 'mandante_Placar', 'visitante_Placar']].dropna()
df_jogos_reais.to_csv("jogos_limpos_para_ia.csv", index=False)

print(f"✅ Histórico reconstruído! Temos {len(df_jogos_reais)} jogos reais na base.")

# ========================================================
# O ALGORITMO DETETIVE: DESCOBRINDO OS JOGOS PENDENTES
# ========================================================
df_2026 = df_jogos_reais[df_jogos_reais['data'].dt.year == 2026]
times_2026 = sorted(list(set(df_2026['mandante']).union(set(df_2026['visitante']))))

print(f"🛡️ {len(times_2026)} times identificados no Brasileirão de 2026.")

# Lista todos os jogos que já ocorreram neste ano
jogos_realizados = set()
for _, row in df_2026.iterrows():
    jogos_realizados.add((row['mandante'], row['visitante']))

# Descobre o que falta: cruza todos contra todos e verifica o que ainda não aconteceu
jogos_pendentes = []
for mandante in times_2026:
    for visitante in times_2026:
        if mandante != visitante:
            if (mandante, visitante) not in jogos_realizados:
                jogos_pendentes.append({"mandante": mandante, "visitante": visitante})

df_pendentes = pd.DataFrame(jogos_pendentes)
df_pendentes.to_csv("jogos_pendentes_2026.csv", index=False)

print(f"⏳ O algoritmo deduziu que faltam exatamente {len(df_pendentes)} jogos para o fim do campeonato.")
print("=" * 60)
print("TUDO PRONTO! Pode rodar a sua Simulação de Monte Carlo novamente.")