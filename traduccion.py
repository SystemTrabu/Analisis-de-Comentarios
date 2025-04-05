from transformers import pipeline
from spanlp.domain.strategies import Preprocessing, NumbersToVowelsInLowerCase
from spanlp.palabrota import Palabrota
from spanlp.domain.countries import Country
from spanlp.domain.strategies import JaccardIndex
import difflib

#pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-es")
#result = pipe("A wasp stung him and he had an allergic reaction.")

#translated_text = result[0]['translation_text']
texto_groseria="hijo de put"
strategies = [NumbersToVowelsInLowerCase()]
preprocessor = Preprocessing(data=texto_groseria, clean_strategies=strategies)
cleaned_text = preprocessor.clean()

print(cleaned_text)




jaccard = JaccardIndex(threshold=0.8, normalize=False, n_gram=1)
print("Parecido en: ",difflib.SequenceMatcher(None, "put", "puto").ratio())
print(f"Parecido span: {jaccard.calculate("put", "puto")}")
palabrota = Palabrota(censor_char="*", countries=[Country.MEXICO], distance_metric=jaccard)
print(palabrota.censor(cleaned_text))



