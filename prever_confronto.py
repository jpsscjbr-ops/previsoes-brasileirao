import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import poisson
import warnings

warnings.filterwarnings('ignore')

def treinar_e_prever(time_casa, time_fora):
    print("🔄 Lendo a base de dados e preparando o modelo...")
    df = pd.read_csv("jogos_limpos_para_ia.csv")
    
    # Ordena cronologicamente
    df['data'] = pd.to_datetime(df['data'])
    df = df.sort_values('data').reset_index(drop=True)
    
    # Calcula a média móvel recente de gols (momentum)
    df_casa_t = df[['data', 'mandante', 'mandante_Placar']].rename(columns={'mandante': 'time', 'mandante_Placar': 'gols'})
    df_fora_t = df[['data', 'visitante', 'visitante_Placar']].rename(columns={'visitante': 'time', 'visitante_Placar': 'gols'})
    df_times = pd.concat([df_casa_t, df_fora_t]).sort_values(by=['time', 'data'])
    
    df_times['media_gols_5'] = df_times.groupby('time')['gols'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    )
    media_geral = df_times['gols'].mean()
    df_times['media_gols_5'] = df_times['media_gols_5'].fillna(media_geral)
    
    # Mescla de volta
    medias_mandante = df_times[['data', 'time', 'media_gols_5']].rename(columns={'time': 'mandante', 'media_gols_5': 'momento_mandante'})
    medias_visitante = df_times[['data', 'time', 'media_gols_5']].rename(columns={'time': 'visitante', 'media_gols_5': 'momento_visitante'})
    
    df = df.merge(medias_mandante, on=['data', 'mandante'], how='left')
    df = df.merge(medias_visitante, on=['data', 'visitante'], how='left')
    
    # Prepara o dataset codificado
    df_modelo = pd.get_dummies(df, columns=['mandante', 'visitante'])
    X = df_modelo.drop(['data', 'mandante_Placar', 'visitante_Placar'], axis=1)
    y_mandante = df['mandante_Placar']
    y_visitante = df['visitante_Placar']
    
    # Treina os modelos de IA
    print("🧠 Treinando os cérebros da IA...")
    mod_casa = RandomForestRegressor(n_estimators=100, random_state=42)
    mod_fora = RandomForestRegressor(n_estimators=100, random_state=42)
    mod_casa.fit(X, y_mandante)
    mod_fora.fit(X, y_visitante)
    
    # Monta a linha de entrada para o jogo desejado
    jogo_futuro = pd.DataFrame(columns=X.columns)
    jogo_futuro.loc[0] = 0 
    
    coluna_casa = f"mandante_{time_casa}"
    coluna_fora = f"visitante_{time_fora}"
    
    if coluna_casa in jogo_futuro.columns: 
        jogo_futuro[coluna_casa] = 1
    else:
        return print(f"❌ Erro: O time mandante '{time_casa}' não foi encontrado na base de dados.")
        
    if coluna_fora in jogo_futuro.columns: 
        jogo_futuro[coluna_fora] = 1
    else:
        return print(f"❌ Erro: O time visitante '{time_fora}' não foi encontrado na base de dados.")
        
    # Previsão de gols pela IA
    gols_casa_previstos = mod_casa.predict(jogo_futuro)[0]
    gols_fora_previstos = mod_fora.predict(jogo_futuro)[0]
    
    # Aplicação de Poisson
    prob_casa, prob_empate, prob_fora = 0.0, 0.0, 0.0
    placar_provavel = ("", 0.0)
    
    for i in range(7):
        for j in range(7):
            p = poisson.pmf(i, gols_casa_previstos) * poisson.pmf(j, gols_fora_previstos)
            if i > j: prob_casa += p
            elif i == j: prob_empate += p
            else: prob_fora += p
            
            if p > placar_provavel[1]:
                placar_provavel = (f"{i}x{j}", p)
                
    # Exibe o relatório final
    print("\n" + "=" * 50)
    print(f"🏟️  CONFRONTO: {time_casa} (Casa) x {time_fora} (Fora)")
    print("=" * 50)
    print(f"📊 Expectativa de Gols (IA)  -> {time_casa}: {gols_casa_previstos:.2f} | {time_fora}: {gols_fora_previstos:.2f}")
    print("-" * 50)
    print(f"🏠 Vitória de {time_casa} : {prob_casa * 100:.1f}%")
    print(f"🤝 Empate               : {prob_empate * 100:.1f}%")
    print(f"✈️  Vitória de {time_fora} : {prob_fora * 100:.1f}%")
    print("-" * 50)
    print(f"🎯 Placar Mais Provável : {placar_provavel[0]} ({placar_provavel[1] * 100:.1f}% de chance)")
    print("=" * 50)

if __name__ == "__main__":
    # Altere os nomes dos times aqui para testar confrontos diferentes
    time_mandante = "Flamengo"
    time_visitante = "Palmeiras"
    
    treinar_e_prever(time_mandante, time_visitante)