import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import poisson
import warnings

warnings.filterwarnings('ignore')

print("🧠 Iniciando o Motor de Expected Goals (xG)...")

# 1. LÊ A BASE DE DADOS OFICIAL QUE BAIXAMOS NA ETAPA 1
df = pd.read_csv("jogos_limpos_para_ia.csv")

# 2. CALCULA AS MÉDIAS DO CAMPEONATO (A BÚSSOLA DA IA)
media_gols_casa = df['mandante_Placar'].mean()
media_gols_fora = df['visitante_Placar'].mean()
print(f"📊 Média do Brasileirão: {media_gols_casa:.2f} gols em casa | {media_gols_fora:.2f} gols fora.")

# 3. ENGENHARIA DE DADOS: ÍNDICES DE FORÇA DE ATAQUE E DEFESA
# Compara o desempenho de cada time com a média geral da liga
forca_ataque_casa = df.groupby('mandante')['mandante_Placar'].mean() / media_gols_casa
forca_defesa_casa = df.groupby('mandante')['visitante_Placar'].mean() / media_gols_fora

forca_ataque_fora = df.groupby('visitante')['visitante_Placar'].mean() / media_gols_fora
forca_defesa_fora = df.groupby('visitante')['mandante_Placar'].mean() / media_gols_casa

# Injeta a inteligência na tabela de jogos
df['ataque_mandante'] = df['mandante'].map(forca_ataque_casa)
df['defesa_mandante'] = df['mandante'].map(forca_defesa_casa)
df['ataque_visitante'] = df['visitante'].map(forca_ataque_fora)
df['defesa_visitante'] = df['visitante'].map(forca_defesa_fora)

print("⚙️ Construindo os cérebros com os novos índices...")

# Traduz os nomes para máquina e adiciona as novas variáveis superpoderosas
df_modelo = pd.get_dummies(df, columns=['mandante', 'visitante'])
X = df_modelo.drop(['data', 'mandante_Placar', 'visitante_Placar'], axis=1)

mod_casa = RandomForestRegressor(n_estimators=100, random_state=42)
mod_fora = RandomForestRegressor(n_estimators=100, random_state=42)
mod_casa.fit(X, df['mandante_Placar'])
mod_fora.fit(X, df['visitante_Placar'])

def prever_jogo_avancado(time_casa, time_fora):
    # Cria uma prancheta do jogo com as métricas do campeonato
    jogo = pd.DataFrame(columns=X.columns)
    jogo.loc[0] = 0 
    
    if f"mandante_{time_casa}" in jogo.columns: jogo[f"mandante_{time_casa}"] = 1
    if f"visitante_{time_fora}" in jogo.columns: jogo[f"visitante_{time_fora}"] = 1
    
    # Injeta a força real calculada dos dois times
    jogo['ataque_mandante'] = forca_ataque_casa.get(time_casa, 1.0)
    jogo['defesa_mandante'] = forca_defesa_casa.get(time_casa, 1.0)
    jogo['ataque_visitante'] = forca_ataque_fora.get(time_fora, 1.0)
    jogo['defesa_visitante'] = forca_defesa_fora.get(time_fora, 1.0)
    
    # IA processa o xG baseado no cruzamento dessas forças
    xg_casa = mod_casa.predict(jogo)[0]
    xg_fora = mod_fora.predict(jogo)[0]
    
    # Poisson calcula as porcentagens exatas
    prob_casa, prob_empate, prob_fora = 0.0, 0.0, 0.0
    for i in range(7):
        for j in range(7):
            p = poisson.pmf(i, xg_casa) * poisson.pmf(j, xg_fora)
            if i > j: prob_casa += p
            elif i == j: prob_empate += p
            else: prob_fora += p
            
    print("\n" + "="*55)
    print(f"🏟️  RAIO-X AVANÇADO: {time_casa} x {time_fora}")
    print("="*55)
    print(f"🔥 Força de Ataque: {time_casa} ({jogo['ataque_mandante'][0]:.2f}) vs {time_fora} ({jogo['ataque_visitante'][0]:.2f})")
    print(f"🛡️  Muro Defensivo: {time_casa} ({jogo['defesa_mandante'][0]:.2f}) vs {time_fora} ({jogo['defesa_visitante'][0]:.2f})")
    print("-" * 55)
    print(f"📊 xG (Gols Esperados) -> {time_casa}: {xg_casa:.2f} | {time_fora}: {xg_fora:.2f}")
    print("-" * 55)
    print(f"🏠 Vitória ({time_casa}) : {prob_casa*100:.1f}%")
    print(f"🤝 Empate           : {prob_empate*100:.1f}%")
    print(f"✈️  Vitória ({time_fora}) : {prob_fora*100:.1f}%")
    print("="*55)

if __name__ == "__main__":
    # Teste o raio-X do xG com um confronto da rodada
    prever_jogo_avancado("Flamengo", "Palmeiras")