import numpy as np
from scipy.stats import poisson

def calcular_probabilidades_partida(gols_casa_previstos, gols_fora_previstos, max_gols=7):
    """
    Aplica a Distribuição de Poisson para calcular as probabilidades 
    de vitória, empate e derrota, além do placar exato mais provável.
    """
    prob_casa_vence = 0.0
    prob_empate = 0.0
    prob_fora_vence = 0.0
    
    placar_mais_provavel = ("", 0.0)
    
    # Simula todas as combinações possíveis de placares (de 0x0 até 6x6)
    for i in range(max_gols):  # Gols do Mandante
        for j in range(max_gols):  # Gols do Visitante
            # Probabilidade independente de cada time marcar 'i' e 'j' gols
            p_casa = poisson.pmf(i, gols_casa_previstos)
            p_fora = poisson.pmf(j, gols_fora_previstos)
            
            # Probabilidade conjunta do placar exato (i x j)
            p_placar = p_casa * p_fora
            
            # Acumula nos resultados do jogo
            if i > j:
                prob_casa_vence += p_placar
            elif i == j:
                prob_empate += p_placar
            else:
                prob_fora_vence += p_placar
                
            # Identifica o placar mais provável
            if p_placar > placar_mais_provavel[1]:
                placar_mais_provavel = (f"{i}x{j}", p_placar)
                
    # Retorna os valores convertidos em porcentagem formatada
    return {
        "vitoria_casa": prob_casa_vence * 100,
        "empate": prob_empate * 100,
        "vitoria_fora": prob_fora_vence * 100,
        "placar_provavel": placar_mais_provavel[0],
        "prob_placar": placar_mais_provavel[1] * 100
    }

# ==========================================
# TESTANDO O MOTOR COM UM PLACAR FICTÍCIO
# ==========================================
if __name__ == "__main__":
    # Digamos que a IA previu que o mandante fará 1.65 gols e o visitante 0.85 gols
    gols_casa_ia = 1.65
    gols_fora_ia = 0.85
    
    resultado = calcular_probabilidades_partida(gols_casa_ia, gols_fora_ia)
    
    print("⚽ SIMULAÇÃO DE PROBABILIDADES (DISTRIBUIÇÃO DE POISSON)")
    print("-" * 55)
    print(f"Gols Previstos pela IA -> Mandante: {gols_casa_ia} | Visitante: {gols_fora_ia}")
    print("-" * 55)
    print(f"🏠 Vitória do Mandante : {resultado['vitoria_casa']:.1f}%")
    print(f"🤝 Empate              : {resultado['empate']:.1f}%")
    print(f"✈️  Vitória do Visitante: {resultado['vitoria_fora']:.1f}%")
    print("-" * 55)
    print(f"🎯 Placar Mais Provável: {resultado['placar_provavel']} ({resultado['prob_placar']:.1f}% de chance)")
    print("-" * 55)