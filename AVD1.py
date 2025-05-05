import itertools
import numpy as np
import pandas as pd

k = int(input("Informe o número de fatores (2 a 5): "))
r = int(input("Informe o número de réplicas (1 a 3): "))


def gerar_tabela_sinais(k):

    niveis = list(itertools.product([-1, 1], repeat=k))
    niveis_corrigido = [tuple(reversed(n)) for n in niveis]
    colunas = [f'F{i+1}' for i in range(k)]
    df = pd.DataFrame(niveis_corrigido, columns=colunas)
    for i in range(2, k + 1):
        for interacao in itertools.combinations(colunas, i):
            nome = ''.join(interacao)
            df[nome] = df[list(interacao)].prod(axis=1)

    return df



def obter_respostas(tabela, r):
    respostas = []
    print("\nInsira os valores de resposta (y), separados por espaço se houver replicações.")
    for i, linha in tabela.iterrows():
        prompt = f"Tratamento {i + 1} {linha.values.tolist()[:k]}: "
        while True:
            try:
                valores = input(prompt).strip().split()
                y_vals = [float(v) for v in valores]
                if len(y_vals) != r:
                    raise ValueError()
                respostas.append(y_vals)
                break
            except:
                print(f"Entrada inválida. Insira exatamente {r} valores numéricos.")
    return np.array(respostas)

def calcular_efeitos(tabela, respostas, k):
    y_medias = respostas.mean(axis=1)
    tabela = tabela.copy()
    tabela['ȳ'] = y_medias

    colunas_efeitos = tabela.columns[:-1]  
    efeitos = {}
    for col in colunas_efeitos:
        efeitos[col] = np.sum(tabela[col] * tabela['ȳ']) / (2 ** k)

    return efeitos, tabela

def calcular_variancia_detalhada(respostas, efeitos, r, k):
    y_medias = respostas.mean(axis=1)
    y_geral = y_medias.mean()

    # Soma Total dos Quadrados (SST) 
    sq_total = np.sum((respostas.flatten() - y_geral) ** 2)

    # Soma dos Quadrados do Erro (SSE) 
    sq_erro = np.sum((y_medias.repeat(r) - respostas.flatten()) ** 2)

    # Soma dos Quadrados por fator/interação (SSFx) 
    ss_fatores = {nome: (2 ** k) * r * (efeito ** 2) for nome, efeito in efeitos.items()}

    # Soma dos Quadrados do Modelo (SSM) 
    sq_modelo = sum(ss_fatores.values())

    # Coeficiente de Determinação (R²) 
    r2 = sq_modelo / sq_total

    print("\n=== Componentes da Decomposição da Variância ===")
    for nome, ss in ss_fatores.items():
        porcentagem = 100 * ss / sq_total  
        print(f"SS{nome} = {ss:.4f} ({porcentagem:.2f}% da variância total)")
    print(f"SSE (Erro) = {sq_erro:.4f}")
    print(f"SST (Total) = {sq_total:.4f}")
    print(f"Coeficiente de Determinação (R²) = {r2:.4%}")
    erro = 1 - r2
    print(f"Erro atribuido = {erro:.4%}")





tabela = gerar_tabela_sinais(k)

respostas = obter_respostas(tabela, r)

efeitos, tabela_com_y = calcular_efeitos(tabela, respostas, k)

print("\n=== Resultados ===")
df_result = pd.DataFrame.from_dict(efeitos, orient='index', columns=['Efeito'])

print("\nTabela de Sinais com ȳ:")
print(tabela_com_y)

calcular_variancia_detalhada(respostas, efeitos, r, k)
