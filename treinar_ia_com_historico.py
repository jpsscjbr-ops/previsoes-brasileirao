import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

print("📖 Lendo a base de dados limpa...")
df = pd.read_csv("jogos_limpos_para_ia.csv")

# Garante que a data está ordenada cronologicamente
df['data'] = pd.to_datetime(df['data'])
df = df.sort_values('data').reset_index(drop=True)

print("📈 Calculando a forma recente (últimos 5 jogos de cada time)...")

# 1. Transformamos o formato de jogos em uma linha do tempo única por time
df_casa = df[['data', 'mandante', 'mandante_Placar']].rename(columns={'mandante': 'time', 'mandante_Placar': 'gols'})
df_fora = df[['data', 'visitante', 'visitante_Placar']].rename(columns={'visitante': 'time', 'visitante_Placar': 'gols'})
df_times = pd.concat([df_casa, df_fora]).sort_values(by=['time', 'data'])

# 2. Média móvel de gols marcados nos últimos 5 jogos (shift(1) garante que só usa o passado!)
df_times['media_gols_5'] = df_times.groupby('time')['gols'].transform(
    lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
)

# Preenche os primeiros jogos de cada time com a média geral do campeonato
media_geral = df_times['gols'].mean()
df_times['media_gols_5'] = df_times['media_gols_5'].fillna(media_geral)

# 3. Devolvemos as médias para a tabela principal de mandantes e visitantes
medias_mandante = df_times[['data', 'time', 'media_gols_5']].rename(columns={'time': 'mandante', 'media_gols_5': 'momento_mandante'})
medias_visitante = df_times[['data', 'time', 'media_gols_5']].rename(columns={'time': 'visitante', 'media_gols_5': 'momento_visitante'})

df = df.merge(medias_mandante, on=['data', 'mandante'], how='left')
df = df.merge(medias_visitante, on=['data', 'visitante'], how='left')

print("⚙️ Preparando matriz de features com o momento dos times...")
df_modelo = pd.get_dummies(df, columns=['mandante', 'visitante'])

# Agora a IA estuda tanto o nome quanto o momento físico recente!
X = df_modelo.drop(['data', 'mandante_Placar', 'visitante_Placar'], axis=1)
y_mandante = df['mandante_Placar']
y_visitante = df['visitante_Placar']

print("🧠 Treinando a IA inteligente com histórico de momentum...")
modelo_mandante = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_visitante = RandomForestRegressor(n_estimators=100, random_state=42)

modelo_mandante.fit(X, y_mandante)
modelo_visitante.fit(X, y_visitante)

print("✅ Modelo aprimorado com sucesso!")
print("-" * 50)