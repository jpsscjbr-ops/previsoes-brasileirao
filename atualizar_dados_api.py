import requests
import pandas as pd
import time

print("🔌 Conectando aos servidores ao vivo da API-Football...")
print("Buscando a realidade exata das temporadas 2024, 2025 e 2026...")

API_KEY = "9931c1241428f009e692821acca124a8"
HEADERS = {
    "x-apisports-key": API_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io"
}

temporadas = [2024, 2025, 2026]
jogos_reais = []
jogos_futuros = []

for ano in temporadas:
    # ID 71 é o Campeonato Brasileiro Série A oficial
    url = f"https://v3.football.api-sports.io/fixtures?league=71&season={ano}"
    resposta = requests.get(url, headers=HEADERS)
    
    if resposta.status_code == 200:
        dados = resposta.json()
        for partida in dados.get("response", []):
            status = partida["fixture"]["status"]["short"]
            
            # Salva os jogos que já aconteceram para treinar a IA
            if status in ["FT", "AET", "PEN"]:
                jogos_reais.append({
                    "data": partida["fixture"]["date"][:10],
                    "mandante": partida["teams"]["home"]["name"],
                    "visitante": partida["teams"]["away"]["name"],
                    "mandante_Placar": partida["goals"]["home"],
                    "visitante_Placar": partida["goals"]["away"]
                })
            # Identifica os jogos que ainda faltam acontecer na temporada atual
            elif ano == 2026 and status in ["TBD", "NS", "PST"]:
                jogos_futuros.append({
                    "mandante": partida["teams"]["home"]["name"],
                    "visitante": partida["teams"]["away"]["name"]
                })
                
        print(f"✅ Temporada {ano} sincronizada com sucesso!")
    else:
        print(f"❌ Erro ao baixar a temporada {ano}.")
        
    # Pausa de 1 segundo exigida pelo plano gratuito para não bloquear a conta
    time.sleep(1)

# Atualiza a base de treinamento da IA com a realidade atual
df_novo = pd.DataFrame(jogos_reais)
df_novo.to_csv("jogos_limpos_para_ia.csv", index=False)

# Cria um arquivo extra mapeando exatamente os jogos que faltam para o Monte Carlo
df_futuro = pd.DataFrame(jogos_futuros)
df_futuro.to_csv("jogos_pendentes_2026.csv", index=False)

print("-" * 50)
print(f"✨ Banco de dados realocado! {len(df_novo)} jogos reais consolidados.")
print(f"Faltam exatamente {len(df_futuro)} jogos para o fim do Brasileirão 2026.")
print("A sua Inteligência Artificial agora enxerga a realidade.")
print("-" * 50)