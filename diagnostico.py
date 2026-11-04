import pandas as pd

print("\n🔍 DIAGNÓSTICO DO BANCO DE DADOS LOCAL")
print("=" * 50)

try:
    df_reais = pd.read_csv("jogos_limpos_para_ia.csv")
    print(f"✅ Arquivo de jogos reais encontrado! Total: {len(df_reais)} partidas.")
    print("🗓️ Últimos 5 jogos registrados na sua máquina:")
    print(df_reais.tail())
except Exception as e:
    print("❌ Erro ao ler jogos reais:", e)

print("-" * 50)

try:
    df_pendentes = pd.read_csv("jogos_pendentes_2026.csv")
    print(f"✅ Arquivo de jogos pendentes encontrado! Faltam: {len(df_pendentes)} partidas.")
    print("⏳ Próximos 5 jogos que a máquina acha que vão acontecer:")
    print(df_pendentes.head())
except Exception as e:
    print("❌ Erro ao ler jogos pendentes:", e)

print("=" * 50)