import time

import numpy as np
from scipy.optimize import differential_evolution
from sklearn.cluster import KMeans


def inercia_vetorizada(populacao, X, k, X_quad, tam_bloco=256):
    """
    Inércia (WCSS) da POPULAÇÃO INTEIRA de uma vez, para usar com
    differential_evolution(..., vectorized=True).

    populacao : array (k*d, S) -- cada COLUNA é um indivíduo (convenção do scipy)
    X         : array (n, d) com as amostras
    k         : número de clusters
    X_quad    : array (n,) com ||x_i||^2, pré-calculado UMA vez
    tam_bloco : nº de indivíduos processados por vez (limita o uso de memória:
                cada bloco aloca tam_bloco * k * n floats)

    Retorna array (S,) com a inércia de cada indivíduo.

    Usa ||x - mu||^2 = ||x||^2 - 2 x.mu + ||mu||^2, de modo que o termo central
    é uma multiplicação de matrizes (BLAS) e o tensor (n, k, d) nunca é criado.
    """
    d = X.shape[1]
    S = populacao.shape[1]
    wcss = np.empty(S)

    for ini in range(0, S, tam_bloco):
        fim = min(ini + tam_bloco, S)
        b = fim - ini

        # b indivíduos x k centroides, cada um com d coordenadas -> (b*k, d)
        mu = populacao[:, ini:fim].T.reshape(b * k, d)

        # Passo 1: distância ao quadrado de cada amostra a cada centroide -> (b, k, n)
        cruzado = (mu @ X.T).reshape(b, k, -1)
        mu_quad = np.sum(mu ** 2, axis=1).reshape(b, k, 1)
        dist_quad = X_quad[None, None, :] - 2 * cruzado + mu_quad

        # Passos 2 e 3: a distância ao centroide do SEU cluster é a mínima sobre k
        wcss[ini:fim] = np.sum(np.min(dist_quad, axis=1), axis=1)

    return wcss


def vencedor(valor_de, valor_base, maior_e_melhor):
    """
    Corrige o 'BEST' do código original: para inércia (e tempo) MENOR é melhor,
    mas para silhouette e V-measure MAIOR é melhor.
    """
    if np.isclose(valor_de, valor_base):
        return "empate"
    de_ganha = valor_de > valor_base if maior_e_melhor else valor_de < valor_base
    return "DE-KM" if de_ganha else "K-MEANS++"


# ---------------- Exemplo de uso ----------------
if __name__ == "__main__":
    from pathlib import Path

    import pandas as pd
    from sklearn.datasets import (
        load_breast_cancer,
        load_digits,
        load_iris,
        load_wine,
        make_blobs,
        make_classification,
    )
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, v_measure_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed=0)

    N_SEMENTES = 10          # execuções independentes de cada método
    SEMENTES = rng.integers(0, 100, size=N_SEMENTES)

    PASTA_BRUTOS = "resultados_brutos"         # um CSV por dataset x método (1 linha por semente)
    ARQUIVO_RESUMO = "resultados_resumo.csv"   # min, média, desvio padrão e máx de cada métrica
    # nome de cada arquivo bruto: <dataset>_<sufixo>.csv  (ex.: iris_dekm.csv)
    SUFIXO_METODO = {
        "DE→KMeans": "dekm",
        "k-means++ (n_init=1)": "kmeanspp_n1",
        "k-means++ (n_init=10)": "kmeanspp_n10",
        "k-means (random, n_init=1)": "random_n1",
    }

    n_samples, k, n_features = 1797, 10, 64
    # X, y = make_blobs(n_samples=n_samples, centers=k, n_features=n_features, random_state=0)

    # nome -> (X, y, k).  Nos datasets reais, k = nº de classes; no sintético, k = 10
    # (como no código original).
    X_iris, y_iris = load_iris(return_X_y=True)
    X_wine, y_wine = load_wine(return_X_y=True)
    X_cancer, y_cancer = load_breast_cancer(return_X_y=True)
    X_digits, y_digits = load_digits(return_X_y=True)
    X_sint, y_sint = make_classification(random_state=0, n_samples=n_samples,
                                         n_features=n_features)
    DATASETS = {
        "iris": (X_iris, y_iris, len(np.unique(y_iris))),
        "wine": (X_wine, y_wine, len(np.unique(y_wine))),
        "breast_cancer": (X_cancer, y_cancer, len(np.unique(y_cancer))),
        "digits": (X_digits, y_digits, len(np.unique(y_digits))),
        "make_classification": (X_sint, y_sint, k),
    }

    # ATENÇÃO: com n_components=None (todas as componentes) o PCA é só uma rotação
    # dos dados, e o K-means é invariante a rotações (diferença de inércia ~1e-12).
    # Só há redução de dimensão se n_components for um inteiro < nº de atributos.
    pca = PCA(
        n_components=None,
        copy=True,
        whiten=False,
        svd_solver='auto',
        tol=0.0,
        iterated_power='auto',
        n_oversamples=10,
        power_iteration_normalizer='auto',
        random_state=0,
    )

    metricas = [("inercia", False), ("silhouette", True), ("v_measure", True),
                ("tempo_s", False)]
    linhas_resumo = []          # alimenta ARQUIVO_RESUMO

    for nome_dataset, (X, y, k) in DATASETS.items():
        n_amostras = X.shape[0]

        t0 = time.perf_counter()
        X = make_pipeline(StandardScaler(), pca).fit_transform(X)
        tempo_preproc = time.perf_counter() - t0
        d = X.shape[1]
        X_quad = np.sum(X ** 2, axis=1) # pré-calculado uma vez para a função vetorizada

        print(f"\n################ {nome_dataset} | n={n_amostras}, d={d}, k={k} "
              f"(D = k*d = {k * d}) ################")

        # Limites: cada coordenada de cada centroide fica dentro do intervalo dos dados
        limites = [(X[:, l].min(), X[:, l].max()) for _ in range(k) for l in range(d)]
        POPSIZE = 15
        
        # n_livres = sum(lo != hi for lo, hi in limites)     # mesma comparação exata do SciPy
        # print(f"D = {len(limites)} | variáveis livres = {n_livres} | N_p esperado = {POPSIZE * n_livres}")

        resultados = {"DE→KMeans": [], "k-means++ (n_init=1)": [],
                      "k-means++ (n_init=10)": [], "k-means (random, n_init=1)": []}

        for semente in SEMENTES:
            # ---- DE (com a função vetorizada) ----
            t0 = time.perf_counter()
            resultado = differential_evolution(
                inercia_vetorizada,
                bounds=limites,
                args=(X, k, X_quad),
                rng=np.random.default_rng(semente),   # semente DIFERENTE por execução
                maxiter=100,
                tol=0,
                mutation=(0.5, 1),
                recombination=0.7,
                strategy='best1bin',
                polish=False,
                disp=True,
                vectorized=True,      # a função recebe a população inteira, shape (k*d, S)
                updating="deferred",  # exigido por vectorized=True
                # workers fica no padrão (1): workers != 1 DESLIGA a vetorização no scipy
            )

            print(resultado.population.shape)

            tempo_de = time.perf_counter() - t0
            centroids = resultado.x.reshape(k, d)

            # ---- KMeans partindo dos centroides do DE (tempo = DE + refino) ----
            t0 = time.perf_counter()
            km_de = KMeans(n_clusters=k, init=centroids, n_init=1, algorithm='lloyd').fit(X)
            tempo_refino = time.perf_counter() - t0

            # ---- Baselines (tempo medido em cada fit) ----
            t0 = time.perf_counter()
            # random_state fixo por semente => resultado reprodutível
            km_1 = KMeans(n_clusters=k, init='k-means++', n_init=1,
                          algorithm='lloyd', random_state=semente).fit(X)
            tempo_km_1 = time.perf_counter() - t0

            t0 = time.perf_counter()
            km_10 = KMeans(n_clusters=k, init='k-means++', n_init=10,
                           algorithm='lloyd', random_state=semente).fit(X)
            tempo_km_10 = time.perf_counter() - t0

            t0 = time.perf_counter()
            # centroides iniciais = k amostras sorteadas ao acaso (sem k-means++)
            km_random = KMeans(n_clusters=k, init='random', n_init=1,
                               algorithm='lloyd', random_state=semente).fit(X)
            tempo_km_random = time.perf_counter() - t0

            modelos = {
                "DE→KMeans": (km_de, tempo_de + tempo_refino),
                "k-means++ (n_init=1)": (km_1, tempo_km_1),
                "k-means++ (n_init=10)": (km_10, tempo_km_10),
                "k-means (random, n_init=1)": (km_random, tempo_km_random),
            }
            for nome, (km, tempo) in modelos.items():
                resultados[nome].append({
                    "semente": int(semente),
                    "inercia": km.inertia_,
                    "silhouette": silhouette_score(X, km.labels_),
                    "v_measure": v_measure_score(y, km.labels_),   # (verdadeiro, previsto)
                    "tempo_s": tempo,
                })
            # detalhamento do tempo do DE→KMeans
            resultados["DE→KMeans"][-1]["tempo_de_s"] = tempo_de
            resultados["DE→KMeans"][-1]["tempo_refino_s"] = tempo_refino
            resultados["DE→KMeans"][-1]["inercia_de"] = resultado.fun      # antes do refino
            resultados["DE→KMeans"][-1]["geracoes_de"] = resultado.nit

        # ---------------- Resumo (deste dataset) ----------------
        baselines = [nome for nome in resultados if nome != "DE→KMeans"]
        for metrica, maior_e_melhor in metricas:
            placares = {}
            for base in baselines:
                placar = {"DE-KM": 0, "K-MEANS++": 0, "empate": 0}
                for a, b in zip(resultados["DE→KMeans"], resultados[base]):
                    placar[vencedor(a[metrica], b[metrica], maior_e_melhor)] += 1
                placares[base] = placar

            print(f"\n=== {nome_dataset} | {metrica} ({'maior' if maior_e_melhor else 'menor'} "
                  f"é melhor) | {N_SEMENTES} sementes ===")
            print(f"{'método':<28}{'min':>13}{'mediana':>13}{'max':>13}")
            for nome, lista in resultados.items():
                v = np.array([r[metrica] for r in lista])
                print(f"{nome:<28}{v.min():>13.4f}{np.median(v):>13.4f}{v.max():>13.4f}")

            for base in baselines:
                print(f"  DE→KMeans vs {base}: {placares[base]}")

            if metrica == "tempo_s":
                t_de = np.median([r["tempo_de_s"] for r in resultados["DE→KMeans"]])
                t_ref = np.median([r["tempo_refino_s"] for r in resultados["DE→KMeans"]])
                print(f"  (DE→KMeans, medianas: DE = {t_de:.3f} s + refino KMeans = {t_ref:.3f} s)")

        # ---------------- CSVs ----------------
        # (a) dados brutos: um arquivo por dataset x método, 1 linha por semente
        pasta = Path(PASTA_BRUTOS)
        pasta.mkdir(parents=True, exist_ok=True)
        for nome, lista in resultados.items():
            brutos = pd.DataFrame([{"dataset": nome_dataset, "n_amostras": n_amostras,
                                    "n_dim": d, "k": k, "metodo": nome, **r} for r in lista])
            brutos = brutos.dropna(axis=1, how="all")   # só o DE→KMeans tem colunas do DE
            brutos.to_csv(pasta / f"{nome_dataset}_{SUFIXO_METODO[nome]}.csv", index=False)

        # (b) resumo geral: min, média, desvio padrão (amostral, ddof=1) e máx por métrica
        for nome, lista in resultados.items():
            for metrica, _ in metricas:
                v = pd.Series([r[metrica] for r in lista])
                linhas_resumo.append({
                    "dataset": nome_dataset, "n_amostras": n_amostras, "n_dim": d, "k": k,
                    "metodo": nome, "metrica": metrica, "n_sementes": len(v),
                    "min": v.min(), "media": v.mean(), "desvio_padrao": v.std(), "max": v.max(),
                })
        # salvo a cada dataset: se a execução for interrompida, o que já rodou fica nos arquivos
        pd.DataFrame(linhas_resumo).to_csv(ARQUIVO_RESUMO, index=False)

    print(f"\nDados brutos em '{PASTA_BRUTOS}/' (um CSV por dataset x método) e "
          f"resumo geral em '{ARQUIVO_RESUMO}'.")