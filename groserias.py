from spanlp.palabrota import Palabrota
from spanlp.domain.countries import Country
from spanlp.domain.strategies import JaccardIndex

def censurar_groseria(texto):
    print(f"Texto en censurar groseria {texto}")
    jaccard = JaccardIndex(threshold=0.9, normalize=False, n_gram=1)
    palabrota = Palabrota(censor_char="*", countries=[Country.MEXICO], distance_metric=jaccard, exclude=["premio","precio", "medalla"])
    resultado=palabrota.censor(texto)
    print(f"Estoy devolviendo: {resultado}")
    return resultado