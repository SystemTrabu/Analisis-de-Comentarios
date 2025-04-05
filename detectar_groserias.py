import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import re
import unicodedata

# Función de limpieza de texto 
def limpiar_texto(texto):
    if not isinstance(texto, str):
        return ""
    
    texto = texto.lower()
    
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    
    texto = re.sub(r'https?://\S+|www\.\S+', ' ', texto)
    
    texto = re.sub(r'@\w+', '@usuario', texto)
    texto = re.sub(r'#\w+', '#hashtag', texto)
    
    texto = re.sub(r'([a-z])\1{2,}', r'\1\1', texto)  # Reduce repeticiones a máximo 2
    
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    texto = re.sub(r'([!?.,;:])\1+', r'\1', texto)
    
    return texto

ruta_modelo = "./detector_groserias_final"
tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
model = AutoModelForSequenceClassification.from_pretrained(ruta_modelo)

# Función para usar el modelo entrenado
def detectar_groserías(texto):
    texto_limpio = limpiar_texto(texto)
    
    inputs = tokenizer(texto_limpio, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    
    return {
        "texto_original": texto,
        "texto_procesado": texto_limpio,
        "es_groseria": bool(probs.argmax().item()),
        "probabilidad": probs[0][1].item()
    }

if __name__ == "__main__":
    textos_prueba = [
        "Info",
        "buenas tardes tendrá algún número para mandarle inf sin compromiso",
        "Con esa carita que no mataba una mosca , detrás de él había una caja de Pandora, realmente los idol o celebridades ,esconden su verdadera personalidad ,y se muestran disque callados y tímidos en la pantalla. Hipocresía en su máxima expresión.",
        "Por eso chicas no elijan a una pareja solo por su cara bonita, primero conozcan bien al chico asegúrense de que sea buena persona que es lo más importante .",
        "Ridículas, actúan igual que esa gente machista que dice que las niñas provocan a la gente por vestirse de cierta forma, por provocarlos, que de seguro se escapó con él toda esa gente que comenta como cuando una niña desaparece así de ridículas se ven defendiendo a ESE PEDOFILO",
        "Que salga él de una vez a disculparse x ocultar su relación y como la negó el año pasado. Sino van a salir mas cosas intimas de ellos.",
        "Los ilustradores desempleados",
        "Si saben que el agua no se acaba verdad 🙃Es imposible que se gaste agua, ya que tiene un ciclo infinito y el agua es imposible de destruir. Incluso en contaminación se vuelve potable bajo el ciclo natural.",
        "Fuera de joda es muy interesante que tan rápido se tiene que ser para que la percepción del tiempo sea diferente a la de un humano normal",
        "Tengo un jale el sábado me prestan una raptor",
        "Viendo la porquería de atención y como fallan los coches de Ford Jalisco, la marca china MG no se ve tan culera 🤣🤦",
        "Dan risa igual que el abogado pendejete",
        "Saquen las trocas prestadas para dar el rol RATEROS",
        "Que pésimo servicio, rateros 😡",
        "E ahi donde se rompen las reglas de todas las compañías automotrices con los problemas de los talleres de servicios del cual el sacar sin permiso los carros a pero no lleves tú el carro a taller ajeno te la hacen de pedo de que pierdes las garantías así como se les puede confiar",
        "justicia para RUDY",
        "Soy mamá de un hijo con autismo y uno con tdha Justicia para Rudy y cárcel para sus 5 agresores",
        "Coloque todo eso y fue horrible",
        "Excelente trabajo maestra Fatima PM le deseo lo mejor en lo que venga y sobre todo que vengan más éxitos en lo personal como siempre le mando un fuerte abrazo.",
        "“¿puedes o no, mediocre?” Es mi mantra de este año."

    ]
    
    print("Analizando textos...")
    for texto in textos_prueba:
        resultado = detectar_groserías(texto)
        emoji = "🚫" if resultado["es_groseria"] else "✅"
        print(f"\n{emoji} Texto: {resultado['texto_original']}")
        print(f"   Texto limpio: {resultado['texto_procesado']}")
        print(f"   Groseria: {resultado['es_groseria']} (Probabilidad: {resultado['probabilidad']:.2%})")


