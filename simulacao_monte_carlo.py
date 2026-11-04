import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import poisson
import warnings

warnings.filterwarnings('ignore')

def executar_monte_carlo_com_xg(num_simulacoes=5000):
    print("📖 Lendo a tabela real e preparando o terreno...")
    df_hist = pd.read_csv("jogos_limpos_para_ia.csv")
    df_pendentes = pd.read_csv("jogos_pendentes_2026.csv")
    
    df_hist['data'] = pd.to_datetime(df_hist['data'])
    df_2026 = df_hist[df_hist['data'].dt.year == 2026]
    
    times_2026 = sorted(list(set(df_2026['mandante']).union(set(df_2026['visitante'])).union(set(df_pendentes['mandante'])).union(set(df_pendentes['visitante']))))
    
    # 1. PONTUAÇÃO REAL DA TABELA
    pontos = {t: 0 for t in times_2026}
    saldo = {t: 0 for t in times_2026}
    gp = {t: 0 for t in times_2026}
    
    for _, row in df_2026.iterrows():
        m, v = row['mandante'], row['visitante']
        gc, gf = int(row['mandante_Placar']), int(row['visitante_Placar'])
        
        gp[m] += gc
        gp[v] += gf
        saldo[m] += (gc - gf)
        saldo[v] += (gf - gc)
        
        if gc > gf: pontos[m] += 3
        elif gc == gf: 
            pontos[m] += 1
            pontos[v] += 1
        else: pontos[v] += 3

    # ========================================================
    # 2. O CÉREBRO DE xG: CALCULANDO ÍNDICES DE FORÇA
    # ========================================================
    print("⚙️ Calculando os Índices de Força Ofensiva e Defensiva...")
    media_gols_casa = df_hist['mandante_Placar'].mean()
    media_gols_fora = df_hist['visitante_Placar'].mean()
    
    forca_ataque_casa = df_hist.groupby('mandante')['mandante_Placar'].mean() / media_gols_casa
    forca_defesa_casa = df_hist.groupby('mandante')['visitante_Placar'].mean() / media_gols_fora
    forca_ataque_fora = df_hist.groupby('visitante')['visitante_Placar'].mean() / media_gols_fora
    forca_defesa_fora = df_hist.groupby('visitante')['mandante_Placar'].mean() / media_gols_casa
    
    df_hist['ataque_mandante'] = df_hist['mandante'].map(forca_ataque_casa).fillna(1.0)
    df_hist['defesa_mandante'] = df_hist['mandante'].map(forca_defesa_casa).fillna(1.0)
    df_hist['ataque_visitante'] = df_hist['visitante'].map(forca_ataque_fora).fillna(1.0)
    df_hist['defesa_visitante'] = df_hist['visitante'].map(forca_defesa_fora).fillna(1.0)

    # 3. TREINAMENTO DA IA COM AS NOVAS MÉTRICAS
    print("🧠 Treinando a Inteligência Artificial turbinada...")
    df_modelo = pd.get_dummies(df_hist, columns=['mandante', 'visitante'])
    X = df_modelo.drop(columns=['data', 'mandante_Placar', 'visitante_Placar'], errors='ignore')
    
    mod_casa = RandomForestRegressor(n_estimators=50, random_state=42)
    mod_fora = RandomForestRegressor(n_estimators=50, random_state=42)
    mod_casa.fit(X, df_hist['mandante_Placar'])
    mod_fora.fit(X, df_hist['visitante_Placar'])

    # 4. GABARITO DE JOGOS RESTANTES COM FORÇA APLICADA
    gabarito_pendentes = []
    linha_base = pd.DataFrame(columns=X.columns)
    linha_base.loc[0] = 0
    
    for _, row in df_pendentes.iterrows():
        m, v = row['mandante'], row['visitante']
        jogo = linha_base.copy()
        
        if f"mandante_{m}" in jogo.columns: jogo[f"mandante_{m}"] = 1
        if f"visitante_{v}" in jogo.columns: jogo[f"visitante_{v}"] = 1
        
        # Injeta o poder tático exato dos times no simulador
        jogo['ataque_mandante'] = forca_ataque_casa.get(m, 1.0)
        jogo['defesa_mandante'] = forca_defesa_casa.get(m, 1.0)
        jogo['ataque_visitante'] = forca_ataque_fora.get(v, 1.0)
        jogo['defesa_visitante'] = forca_defesa_fora.get(v, 1.0)
        
        xg_c = max(0.1, mod_casa.predict(jogo)[0])
        xg_f = max(0.1, mod_fora.predict(jogo)[0])
        gabarito_pendentes.append((m, v, xg_c, xg_f))

    # ========================================================
    # 5. MONTE CARLO - ROLANDO OS DADOS COM PRECISÃO CIRÚRGICA
    # ========================================================
    print(f"🎲 Disparando {num_simulacoes} simulações de universos paralelos...")
    titulos = {t: 0 for t in times_2026}
    g4 = {t: 0 for t in times_2026}
    z4 = {t: 0 for t in times_2026}

    for sim in range(num_simulacoes):
        pts_sim = pontos.copy()
        sg_sim = saldo.copy()
        gp_sim = gp.copy()
        
        for m, v, xg_c, xg_f in gabarito_pendentes:
            gc = np.random.poisson(xg_c)
            gf = np.random.poisson(xg_f)
            
            gp_sim[m] += gc
            gp_sim[v] += gf
            sg_sim[m] += (gc - gf)
            sg_sim[v] += (gf - gc)
            
            if gc > gf: pts_sim[m] += 3
            elif gc == gf: 
                pts_sim[m] += 1
                pts_sim[v] += 1
            else: pts_sim[v] += 3
                
        tabela = sorted(times_2026, key=lambda t: (pts_sim[t], sg_sim[t], gp_sim[t]), reverse=True)
        
        titulos[tabela[0]] += 1
        for t in tabela[:4]: g4[t] += 1
        for t in tabela[-4:]: z4[t] += 1

    # 6. RELATÓRIO FINAL
    resultados = []
    for t in times_2026:
        resultados.append({
            "Time": t,
            "Pts_Atuais": pontos[t],
            "Titulo_%": round((titulos[t]/num_simulacoes)*100, 1),
            "Liberta_%": round((g4[t]/num_simulacoes)*100, 1),
            "Rebaixamento_%": round((z4[t]/num_simulacoes)*100, 1)
        })
        
    df_final = pd.DataFrame(resultados).sort_values(by=["Titulo_%", "Pts_Atuais"], ascending=[False, False]).reset_index(drop=True)
    
    # Salva também um JSON para podermos usar no nosso futuro site web!
    df_final.to_json("previsoes_brasileirao_2026.json", orient="records", indent=4)
    
    print("\n" + "="*80)
    print("🏆 PROJEÇÃO DO BRASILEIRÃO 2026 (IA + EXPECTED GOALS)")
    print("="*80)
    print(df_final.to_string(index=False))
    print("="*80)
    print("✅ Sucesso! O arquivo 'previsoes_brasileirao_2026.json' foi criado para o site.")

if __name__ == "__main__":
    executar_monte_carlo_com_xg(5000)