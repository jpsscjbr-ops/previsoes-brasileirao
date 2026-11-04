import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import warnings

# Tira os avisos amarelos para deixar a tela limpa
warnings.filterwarnings('ignore')

print("📖 Lendo o arquivo 'jogos_limpos_para_ia.csv'...")
df = pd.read_csv("jogos_limpos_para_ia.csv")

print("⚙️ Traduzindo nomes dos times para linguagem de máquina...")
# Isso transforma a coluna 'mandante' e 'visitante' em dezenas de colunas com 0s e 1s
df_modelo = pd.get_dummies(df, columns=['mandante', 'visitante'])

# Separamos o que a IA vai usar para estudar (X) e o que ela tem que adivinhar (Y)
X = df_modelo.drop(['data', 'mandante_Placar', 'visitante_Placar'], axis=1)
y_mandante = df['mandante_Placar']
y_visitante = df['visitante_Placar']

print("🧠 Treinando a Inteligência Artificial (Isso pode levar alguns segundos)...")
modelo_mandante = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_visitante = RandomForestRegressor(n_estimators=100, random_state=42)

modelo_mandante.fit(X, y_mandante)
modelo_visitante.fit(X, y_visitante)

print("✅ Treinamento concluído com sucesso!")
print("-" * 50)

# ==========================================
# TESTANDO A MÁQUINA COM UM JOGO INÉDITO
# ==========================================
def simular_jogo(time_casa, time_fora):
    # Cria uma prancheta em branco com as mesmas marcações do treinamento
    jogo_futuro = pd.DataFrame(columns=X.columns)
    jogo_futuro.loc[0] = 0 
    
    # Marca '1' para dizer quem está jogando
    coluna_casa = f"mandante_{time_casa}"
    coluna_fora = f"visitante_{time_fora}"
    
    if coluna_casa in jogo_futuro.columns: jogo_futuro[coluna_casa] = 1
    if coluna_fora in jogo_futuro.columns: jogo_futuro[coluna_fora] = 1
        
    # Pede para os cérebros adivinharem os gols
    gols_casa = modelo_mandante.predict(jogo_futuro)[0]
    gols_fora = modelo_visitante.predict(jogo_futuro)[0]
    
    print(f"🔮 PREVISÃO DA IA: {time_casa} {gols_casa:.2f} x {gols_fora:.2f} {time_fora}")

# Vamos testar com o Flamengo jogando no Maracanã contra o Palmeiras
# (Os nomes devem ser exatamente iguais aos que estão no banco de dados)
simular_jogo("Flamengo", "Palmeiras")
print("-" * 50)