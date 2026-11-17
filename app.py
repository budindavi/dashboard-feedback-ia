"""Dashboard de análise de feedback de clientes.

Rode no terminal, dentro desta pasta:

    py -3 -m venv .venv
    .venv\\Scripts\\activate
    pip install -r requirements.txt
    streamlit run app.py

No Python 3.15 sem compilador C++, o pip falha ao montar o PyYAML. Antes do
pip install, rode no PowerShell:  $env:PYYAML_FORCE_LIBYAML = "0"

O sentimento combina regras de palavras com um modelo de linguagem que roda
na sua máquina (CPU). As regras decidem quando reconhecem alguma expressão;
o modelo decide os comentários sem nenhuma expressão conhecida. Na primeira
análise o modelo é baixado do Hugging Face (~540 MB) e fica em cache; depois
funciona sem internet. Sem os pacotes de IA, o app usa só as regras.
Os temas de reclamação e o resumo vêm das regras.
"""

from __future__ import annotations

import io
import re
import unicodedata
from collections.abc import Callable
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

LIMIAR_POSITIVO = 0.75
LIMIAR_NEGATIVO = -0.75
MAX_LINHAS = 5000

MODELO_ID = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
ROTULOS_MODELO = {"positive": "Positivo", "neutral": "Neutro", "negative": "Negativo"}
LOTE_MODELO = 32

NEGACAO = {"nao", "nem", "nunca", "jamais", "sem", "mal", "nada", "nenhum", "nenhuma"}
# "não foi ruim" é no máximo morno, não um elogio: negar algo negativo vale pouco.
FATOR_NEGA_POSITIVO = 0.75
FATOR_NEGA_NEGATIVO = 0.3
# "caro cliente", "barata" (inseto ou preço): palavras que mudam de sentido pelo vizinho.
VOCATIVO = {"cliente", "clientes", "amigo", "amiga", "senhor", "senhora", "parceiro"}
ANTES_DE_INSETO = {"uma", "umas", "a", "as", "achei", "vi", "tinha", "com", "tem", "havia"}
MIN_REPETICAO = 2
INTENSIFICADOR = {
    "muito": 1.45,
    "super": 1.5,
    "extremamente": 1.6,
    "bem": 1.2,
    "pouco": 0.55,
    "mega": 1.4,
    "totalmente": 1.3,
    "mais": 1.25,
}
DEPOIS = {"demais": 1.35, "mesmo": 1.15}
CONTRASTE = {"mas", "porem", "contudo", "entretanto"}
CONTRASTE_PAR = {("so", "que"), ("fora", "isso"), ("apesar", "disso")}
PONTE = {
    "de", "da", "do", "das", "dos", "a", "o", "as", "os", "um", "uma",
    "que", "e", "em", "no", "na", "nos", "nas", "para", "por", "com",
    "se", "ao", "aos", "pra", "estava", "tava", "ficou", "era", "foi",
    "veio", "chegou", "esta", "ta", "estao", "sao",
}

BEBIDAS = {
    "cerveja",
    "refrigerante",
    "refri",
    "suco",
    "bebida",
    "drink",
    "agua",
    "vinho",
    "chopp",
    "lata",
}
COMIDAS = {
    "comida",
    "pizza",
    "lanche",
    "hamburguer",
    "hamburger",
    "burger",
    "marmita",
    "prato",
    "massa",
    "sushi",
    "batata",
    "carne",
    "frango",
    "arroz",
    "feijao",
    "salada",
    "sobremesa",
    "molho",
    "esfirra",
    "acai",
    "pastel",
    "coxinha",
    "porcao",
    "pedido",
}

COLUNAS_TEXTO = {
    "comentario",
    "comentarios",
    "review",
    "reviews",
    "feedback",
    "texto",
    "text",
    "comment",
    "comments",
    "mensagem",
    "depoimento",
    "opiniao",
    "descricao",
    "avaliacao",
    "relato",
}
COLUNAS_NOTA = {"nota", "rating", "estrelas", "stars", "score", "avaliacao"}

ACOES = {
    "Tempo de entrega": (
        "Compare o prazo do app com o tempo real do último mês e corrija a promessa. "
        "Se a espera nasce na cozinha, avise o entregador só quando o pedido estiver lacrado."
    ),
    "Comida": (
        "Prove, no pico, os pratos que aparecem nas reclamações e acerte sal, ponto e porção "
        "antes de mexer no cardápio."
    ),
    "Temperatura": (
        "Feche por último o que esfria e não deixe a embalagem esperando no balcão. "
        "No salão, encurte o caminho entre o passe e a mesa."
    ),
    "Atendimento": (
        "Combine uma resposta curta para atraso e pedido errado: reconhecer o erro, "
        "corrigir e dizer o novo prazo. Use a mesma frase no salão e no chat."
    ),
    "Preço": (
        "Olhe os itens chamados de caros. Onde o preço permanecer, deixe porção e "
        "acompanhamento visíveis na descrição."
    ),
    "Pedido": (
        "Confira item por item em voz alta antes de lacrar. Erro repetido costuma ser "
        "adicional, molho ou bebida."
    ),
    "Embalagem": (
        "Troque a embalagem do que vaza e mande molho em pote separado, fechado."
    ),
    "Higiene": (
        "Trate comentário sobre cabelo, inseto ou sujeira no mesmo dia: touca, superfície "
        "limpa e uma passada na linha de montagem antes do próximo pedido."
    ),
    "Bebida": (
        "Geladeira até a saída e bolsa térmica no caminho. Se o cliente disser que a bebida "
        "veio quente, troque sem discutir."
    ),
}

EXEMPLO = """A pizza chegou quente e o entregador foi educado. Recomendo.
Demorou mais de uma hora e a comida chegou fria. Pedido errado, faltou o refrigerante.
Atendimento bom, mas o hambúrguer estava sem gosto e seco.
Preço justo e porção generosa. Voltarei.
A embalagem vazou o molho e sujou a sacola. Fora isso a comida estava boa.
Gostei do salão, mas o garçom foi mal educado.
Comida excelente, no ponto, e a sobremesa estava deliciosa.
Não recomendo. Caro demais para a qualidade e a salada veio murcha.
Comida normal, entrega normal. Sem destaque.
Achei um cabelo no prato. Experiência horrível.
A cerveja veio gelada e o petisco estava saboroso.
Marmita ok, arroz bom, feijão sem sal.
"""

# frase, peso, aspecto (ou None quando o sentimento é geral)
ENTRADAS = [
    ("nao recomendo", -2.2, None),
    ("nao volto", -1.6, None),
    ("nunca mais", -1.8, None),
    ("sem gosto", -1.8, "Comida"),
    ("sem sal", -1.5, "Comida"),
    ("sem tempero", -1.4, "Comida"),
    ("sem graça", -1.3, "Comida"),
    ("pedido errado", -2.0, "Pedido"),
    ("veio errado", -2.0, "Pedido"),
    ("item errado", -1.8, "Pedido"),
    ("nao veio", -1.7, "Pedido"),
    ("item faltando", -1.8, "Pedido"),
    ("nao vale", -1.5, "Preço"),
    ("nao compensa", -1.5, "Preço"),
    ("preco alto", -1.3, "Preço"),
    ("preco justo", 1.0, "Preço"),
    ("preco bom", 1.0, "Preço"),
    ("mal educado", -1.9, "Atendimento"),
    ("mal educada", -1.9, "Atendimento"),
    ("mal atendido", -1.7, "Atendimento"),
    ("mal atendida", -1.7, "Atendimento"),
    ("bem atendido", 1.4, "Atendimento"),
    ("bem atendida", 1.4, "Atendimento"),
    ("atendimento bom", 1.3, "Atendimento"),
    ("atendimento otimo", 1.7, "Atendimento"),
    ("atendimento excelente", 1.8, "Atendimento"),
    ("atendimento ruim", -1.6, "Atendimento"),
    ("atendimento pessimo", -2.0, "Atendimento"),
    ("comida boa", 1.3, "Comida"),
    ("comida otima", 1.7, "Comida"),
    ("comida excelente", 1.8, "Comida"),
    ("comida ruim", -1.6, "Comida"),
    ("comida fria", -1.7, "Temperatura"),
    ("no ponto", 1.4, "Comida"),
    ("bem temperado", 1.3, "Comida"),
    ("bem temperada", 1.3, "Comida"),
    ("porcao generosa", 1.2, "Comida"),
    ("mais de uma hora", -1.8, "Tempo de entrega"),
    ("hora e meia", -1.7, "Tempo de entrega"),
    ("entrega rapida", 1.3, "Tempo de entrega"),
    ("entrega lenta", -1.4, "Tempo de entrega"),
    ("chegou rapido", 1.2, "Tempo de entrega"),
    ("antes do prazo", 1.3, "Tempo de entrega"),
    ("muita espera", -1.5, "Tempo de entrega"),
    ("um pelo", -1.8, "Higiene"),
    ("sem lacre", -1.4, "Embalagem"),
    ("lacre rompido", -1.5, "Embalagem"),
    ("vazou o molho", -1.7, "Embalagem"),
    ("excelente", 1.9, None),
    ("otimo", 1.7, None),
    ("otima", 1.7, None),
    ("maravilhoso", 1.8, None),
    ("maravilhosa", 1.8, None),
    ("perfeito", 1.6, None),
    ("perfeita", 1.6, None),
    ("delicioso", 1.7, "Comida"),
    ("deliciosa", 1.7, "Comida"),
    ("saboroso", 1.4, "Comida"),
    ("saborosa", 1.4, "Comida"),
    ("gostoso", 1.3, "Comida"),
    ("gostosa", 1.3, "Comida"),
    ("gostei", 1.2, None),
    ("amei", 1.6, None),
    ("adorei", 1.6, None),
    ("recomendo", 1.5, None),
    ("voltarei", 1.4, None),
    ("caprichado", 1.2, None),
    ("caprichada", 1.2, None),
    ("impecavel", 1.6, None),
    ("fresco", 1.0, "Comida"),
    ("fresca", 1.0, "Comida"),
    ("horrivel", -2.0, None),
    ("pessimo", -2.0, None),
    ("pessima", -2.0, None),
    ("ruim", -1.4, None),
    ("decepcionante", -1.7, None),
    ("decepcao", -1.5, None),
    ("odiei", -1.8, None),
    ("lixo", -1.8, None),
    ("atrasou", -1.8, "Tempo de entrega"),
    ("atrasaram", -1.8, "Tempo de entrega"),
    ("atrasado", -1.6, "Tempo de entrega"),
    ("atrasada", -1.6, "Tempo de entrega"),
    ("atraso", -1.6, "Tempo de entrega"),
    ("demorou", -1.6, "Tempo de entrega"),
    ("demora", -1.3, "Tempo de entrega"),
    ("lentidao", -1.3, "Tempo de entrega"),
    ("lento", -1.1, "Tempo de entrega"),
    ("lenta", -1.1, "Tempo de entrega"),
    ("espera", -0.9, "Tempo de entrega"),
    ("pontual", 1.2, "Tempo de entrega"),
    ("rapido", 1.0, "Tempo de entrega"),
    ("rapida", 1.0, "Tempo de entrega"),
    ("frio", -1.5, "Temperatura"),
    ("fria", -1.5, "Temperatura"),
    ("morno", -1.3, "Temperatura"),
    ("morna", -1.3, "Temperatura"),
    ("gelado", 1.1, "Bebida"),
    ("gelada", 1.1, "Bebida"),
    ("quente", 0.9, "Temperatura"),
    ("quentinho", 1.2, "Temperatura"),
    ("quentinha", 1.2, "Temperatura"),
    ("seco", -1.2, "Comida"),
    ("seca", -1.2, "Comida"),
    ("borrachudo", -1.4, "Comida"),
    ("borrachuda", -1.4, "Comida"),
    ("queimado", -1.4, "Comida"),
    ("queimada", -1.4, "Comida"),
    ("cru", -1.5, "Comida"),
    ("crua", -1.5, "Comida"),
    ("oleoso", -1.1, "Comida"),
    ("gorduroso", -1.1, "Comida"),
    ("insosso", -1.3, "Comida"),
    ("salgado", -1.2, "Comida"),
    ("salgada", -1.2, "Comida"),
    ("azedo", -1.3, "Comida"),
    ("estragado", -1.8, "Comida"),
    ("estragada", -1.8, "Comida"),
    ("murcho", -1.3, "Comida"),
    ("murcha", -1.3, "Comida"),
    ("atencioso", 1.3, "Atendimento"),
    ("atenciosa", 1.3, "Atendimento"),
    ("educado", 1.1, "Atendimento"),
    ("educada", 1.1, "Atendimento"),
    ("simpatico", 1.1, "Atendimento"),
    ("simpatica", 1.1, "Atendimento"),
    ("cordial", 1.0, "Atendimento"),
    ("rude", -1.6, "Atendimento"),
    ("grosseiro", -1.6, "Atendimento"),
    ("grosseira", -1.6, "Atendimento"),
    ("ignorou", -1.5, "Atendimento"),
    ("descaso", -1.6, "Atendimento"),
    ("caro", -1.3, "Preço"),
    ("carissimo", -1.7, "Preço"),
    ("carissima", -1.7, "Preço"),
    ("abusivo", -1.5, "Preço"),
    ("barato", 0.9, "Preço"),
    ("faltou", -1.7, "Pedido"),
    ("faltando", -1.6, "Pedido"),
    ("incompleto", -1.5, "Pedido"),
    ("incompleta", -1.5, "Pedido"),
    ("trocado", -1.5, "Pedido"),
    ("trocada", -1.5, "Pedido"),
    ("vazou", -1.6, "Embalagem"),
    ("derramou", -1.6, "Embalagem"),
    ("amassada", -1.3, "Embalagem"),
    ("amassado", -1.3, "Embalagem"),
    ("cabelo", -1.9, "Higiene"),
    ("barata", -2.0, "Higiene"),
    ("mosca", -1.8, "Higiene"),
    ("inseto", -1.8, "Higiene"),
    ("sujo", -1.5, "Higiene"),
    ("suja", -1.5, "Higiene"),
    ("fedido", -1.6, "Higiene"),
    ("fedida", -1.6, "Higiene"),
    ("ok", 0.25, None),
    ("razoavel", 0.3, None),
    ("bom", 1.1, None),
    ("boa", 1.1, None),
    ("generosa", 1.0, "Comida"),
    ("generoso", 1.0, "Comida"),
]


def dobrar(texto: str) -> str:
    """Minúsculas, sem acento, para casar o vocabulário."""
    base = unicodedata.normalize("NFD", str(texto).lower().replace("ç", "c"))
    sem_acento = "".join(ch for ch in base if unicodedata.category(ch) != "Mn")
    return sem_acento


def tokenizar(texto: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", texto)


def compilar(entradas: list[tuple[str, float, str | None]]) -> dict[tuple[str, ...], tuple[float, str | None]]:
    indice: dict[tuple[str, ...], tuple[float, str | None]] = {}
    for frase, peso, aspecto in entradas:
        chave = tuple(tokenizar(dobrar(frase)))
        if chave:
            indice[chave] = (peso, aspecto)
    return indice


INDICE = compilar(ENTRADAS)
TAMANHO_MAX = max(len(chave) for chave in INDICE)


def casar(tokens: list[str], inicio: int) -> tuple[int, float, str | None] | None:
    restante = len(tokens) - inicio
    for tamanho in range(min(TAMANHO_MAX, restante), 0, -1):
        item = INDICE.get(tuple(tokens[inicio : inicio + tamanho]))
        if item is not None:
            peso, aspecto = item
            return tamanho, peso, aspecto
    return None


def distancia(tokens: list[str], inicio: int, tamanho: int, vocabulario: set[str]) -> float | None:
    centro = inicio + (tamanho - 1) / 2
    achadas = [abs(pos - centro) for pos, tok in enumerate(tokens) if tok in vocabulario]
    if not achadas:
        return None
    return min(achadas)


def ajustar_contexto(
    frase: tuple[str, ...],
    peso: float,
    aspecto: str | None,
    tokens: list[str],
    inicio: int,
    tamanho: int,
) -> tuple[float, str | None]:
    chave = " ".join(frase)
    dist_comida = distancia(tokens, inicio, tamanho, COMIDAS)
    dist_bebida = distancia(tokens, inicio, tamanho, BEBIDAS)

    if chave in {"frio", "fria", "morno", "morna"}:
        if dist_bebida is not None and (dist_comida is None or dist_bebida < dist_comida):
            return 0.8, "Bebida"
        return peso, "Temperatura"

    if chave in {"gelado", "gelada"}:
        if dist_comida is not None and (dist_bebida is None or dist_comida < dist_bebida):
            return -1.4, "Temperatura"
        return 1.1, "Bebida"

    if chave in {"quente", "quentinho", "quentinha"}:
        if dist_bebida is not None and (dist_comida is None or dist_bebida < dist_comida):
            return -1.3, "Bebida"
        return max(peso, 0.9), "Temperatura"

    if chave == "barata":
        anterior = tokens[inicio - 1] if inicio > 0 else ""
        if anterior in ANTES_DE_INSETO:
            return peso, "Higiene"
        perto_de_comida = dist_comida is not None and dist_comida <= 4
        if perto_de_comida or anterior in PONTE or anterior in INTENSIFICADOR:
            return 0.9, "Preço"
        return peso, "Higiene"

    if chave == "caro":
        seguinte = tokens[inicio + tamanho] if inicio + tamanho < len(tokens) else ""
        if seguinte in VOCATIVO:
            return 0.0, None

    return peso, aspecto


def aplicar_contexto_local(peso: float, tokens: list[str], inicio: int, tamanho: int) -> float:
    """Negação e intensificador valem só até a palavra anterior de verdade.

    Assim, o "não" de "não recomendo" não inverte um "caro" que vem depois.
    Negar um elogio ("não é bom") vira crítica; negar uma crítica ("não foi
    ruim") vira só um leve alívio, nunca um elogio.
    """
    anteriores = tokens[max(0, inicio - 4) : inicio]
    negado = False
    intens = 1.0
    for tok in reversed(anteriores):
        if tok in INTENSIFICADOR:
            if intens == 1.0:
                intens = INTENSIFICADOR[tok]
            continue
        if tok in PONTE:
            continue
        if tok in NEGACAO:
            negado = True
        break
    if negado:
        intens = -(FATOR_NEGA_POSITIVO if peso > 0 else FATOR_NEGA_NEGATIVO)
    seguinte = inicio + tamanho
    if seguinte < len(tokens) and tokens[seguinte] in DEPOIS:
        intens *= DEPOIS[tokens[seguinte]]
    return peso * intens


def segmentos(tokens: list[str]) -> list[list[str]]:
    partes: list[list[str]] = []
    atual: list[str] = []
    i = 0
    while i < len(tokens):
        salto = 2 if tuple(tokens[i : i + 2]) in CONTRASTE_PAR else 0
        if salto == 0 and tokens[i] in CONTRASTE:
            salto = 1
        if salto:
            if atual:
                partes.append(atual)
            atual = []
            i += salto
            continue
        atual.append(tokens[i])
        i += 1
    if atual:
        partes.append(atual)
    return partes


def ler_trecho(tokens: list[str]) -> list[dict]:
    achados = []
    i = 0
    while i < len(tokens):
        casado = casar(tokens, i)
        if casado is None:
            i += 1
            continue
        tamanho, peso, aspecto = casado
        frase = tuple(tokens[i : i + tamanho])
        peso, aspecto = ajustar_contexto(frase, peso, aspecto, tokens, i, tamanho)
        peso_final = aplicar_contexto_local(peso, tokens, i, tamanho)
        if peso_final != 0:
            achados.append({"peso": peso_final, "aspecto": aspecto})
        i += tamanho
    return achados


def classificar(texto: str) -> dict:
    """Classifica um comentário em Positivo, Neutro ou Negativo.

    O peso de cada expressão muda com negação ("não", "sem", "nunca"),
    intensificador ("muito", "pouco") e contraste ("mas"): o trecho final pesa mais.
    """
    partes = segmentos(tokenizar(dobrar(texto)))
    if not partes:
        return {
            "sentimento": "Neutro",
            "score": 0.0,
            "sinais": 0,
            "dores": [],
            "elogios": [],
            "peso_dor": {},
            "peso_elogio": {},
        }

    score = 0.0
    sinais = 0
    peso_dor: dict[str, float] = {}
    peso_elogio: dict[str, float] = {}
    varios = len(partes) > 1
    for indice, parte in enumerate(partes):
        peso_trecho = 1.0
        if varios:
            peso_trecho = 1.3 if indice == len(partes) - 1 else 0.7
        for achado in ler_trecho(parte):
            sinais += 1
            score += achado["peso"] * peso_trecho
            aspecto = achado["aspecto"]
            if not aspecto:
                continue
            if achado["peso"] <= -0.8:
                peso_dor[aspecto] = min(peso_dor.get(aspecto, 0.0), achado["peso"])
            elif achado["peso"] >= 0.8:
                peso_elogio[aspecto] = max(peso_elogio.get(aspecto, 0.0), achado["peso"])

    if score >= LIMIAR_POSITIVO:
        sentimento = "Positivo"
    elif score <= LIMIAR_NEGATIVO:
        sentimento = "Negativo"
    else:
        sentimento = "Neutro"

    return {
        "sentimento": sentimento,
        "score": round(score, 2),
        "sinais": sinais,
        "dores": list(peso_dor),
        "elogios": list(peso_elogio),
        "peso_dor": peso_dor,
        "peso_elogio": peso_elogio,
    }


def ler_arquivo(arquivo) -> pd.DataFrame:
    nome = arquivo.name.lower()
    if nome.endswith(".csv"):
        bruto = arquivo.getvalue()
        ultimo_erro: Exception | None = None
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(bruto), sep=None, engine="python", encoding=encoding)
            except UnicodeDecodeError as exc:
                ultimo_erro = exc
        raise ValueError("Não consegui ler o CSV. Salve o arquivo em UTF-8.") from ultimo_erro
    if nome.endswith(".xlsx"):
        return pd.read_excel(arquivo, engine="openpyxl")
    raise ValueError("Envie um arquivo .csv ou .xlsx.")


def parece_numerica(serie: pd.Series) -> bool:
    valores = serie.dropna()
    if valores.empty:
        return False
    return pd.to_numeric(valores, errors="coerce").notna().mean() >= 0.6


def sugerir_coluna(tabela: pd.DataFrame) -> str:
    for coluna in tabela.columns:
        # "avaliação" pode ser o texto ou a nota; só vale como texto se não for número.
        if dobrar(str(coluna)).replace(" ", "") in COLUNAS_TEXTO and not parece_numerica(tabela[coluna]):
            return str(coluna)
    # is_string_dtype cobre object, "string" e o "str" padrão do pandas 3.
    textos = [
        col
        for col in tabela.columns
        if pd.api.types.is_string_dtype(tabela[col]) and not parece_numerica(tabela[col])
    ]
    if textos:
        medias = {col: tabela[col].astype(str).str.len().mean() for col in textos}
        return str(max(medias, key=medias.get))
    return str(tabela.columns[0])


def coluna_de_nota(tabela: pd.DataFrame, coluna_texto: str) -> str | None:
    for coluna in tabela.columns:
        if str(coluna) == coluna_texto:
            continue
        if dobrar(str(coluna)).replace(" ", "") not in COLUNAS_NOTA:
            continue
        numerica = pd.to_numeric(tabela[coluna], errors="coerce")
        if numerica.notna().mean() >= 0.6:
            return str(coluna)
    return None


def comentarios_colados(texto: str) -> list[str]:
    blocos = [bloco.strip() for bloco in re.split(r"\n\s*\n", texto) if bloco.strip()]
    if len(blocos) <= 1:
        linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]
        if len(linhas) > 1:
            return linhas
    return blocos


@st.cache_resource(show_spinner="Carregando o modelo de IA. Na primeira vez ele é baixado (~540 MB).")
def _carregar_modelo():
    from transformers import pipeline

    return pipeline("text-classification", model=MODELO_ID, top_k=None, device=-1)


def carregar_modelo() -> tuple[object | None, str | None]:
    """Devolve (modelo, None) ou (None, motivo). Falhas não ficam em cache."""
    try:
        return _carregar_modelo(), None
    except ImportError:
        return None, "Os pacotes de IA (transformers e torch) não estão instalados."
    except Exception as exc:  # rede, disco, modelo corrompido
        return None, f"Não consegui carregar o modelo de IA ({type(exc).__name__}: {exc})."


def sentimento_por_modelo(
    textos: list[str],
    modelo,
    ao_progredir: Callable[[float], None] | None = None,
) -> list[dict]:
    """Classifica em lotes. Score = P(positivo) − P(negativo), de -1 a 1."""
    saida: list[dict] = []
    for inicio in range(0, len(textos), LOTE_MODELO):
        lote = textos[inicio : inicio + LOTE_MODELO]
        respostas = modelo(lote, truncation=True, max_length=512, batch_size=LOTE_MODELO)
        for resposta in respostas:
            probs = {ROTULOS_MODELO.get(item["label"].lower(), item["label"]): item["score"] for item in resposta}
            sentimento = max(probs, key=probs.get)
            saida.append(
                {
                    "sentimento": sentimento,
                    "score": round(probs.get("Positivo", 0.0) - probs.get("Negativo", 0.0), 2),
                    "confianca": round(probs[sentimento], 2),
                }
            )
        if ao_progredir:
            ao_progredir(min(1.0, (inicio + len(lote)) / len(textos)))
    return saida


def analisar_tabela(
    tabela: pd.DataFrame,
    coluna: str,
    coluna_nota: str | None,
    modelo=None,
    ao_progredir: Callable[[float], None] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    avisos: list[str] = []
    colunas = [coluna] + ([coluna_nota] if coluna_nota else [])
    trabalho = tabela.loc[:, colunas].copy()
    trabalho[coluna] = trabalho[coluna].map(lambda valor: "" if pd.isna(valor) else str(valor).strip())
    trabalho = trabalho[trabalho[coluna].ne("") & trabalho[coluna].str.lower().ne("nan")]
    if len(trabalho) > MAX_LINHAS:
        avisos.append(f"Li os primeiros {MAX_LINHAS} comentários. O arquivo tem {len(trabalho)}.")
        trabalho = trabalho.head(MAX_LINHAS)
    if trabalho.empty:
        return pd.DataFrame(), avisos

    # As regras sempre rodam: são elas que acham o tema (dores e elogios).
    extra = pd.DataFrame(trabalho[coluna].map(classificar).tolist(), index=trabalho.index)
    extra["fonte"] = "Regras"
    if modelo is not None:
        # Nos testes, os modelos leves erraram mais que as regras quando elas
        # reconhecem alguma expressão ("péssimo" virou positivo). Por isso o
        # modelo decide só os comentários em que as regras não acharam nada.
        sem_sinal = extra.index[extra["sinais"] == 0]
        if len(sem_sinal):
            ia = pd.DataFrame(
                sentimento_por_modelo(trabalho.loc[sem_sinal, coluna].tolist(), modelo, ao_progredir),
                index=sem_sinal,
            )
            extra.loc[sem_sinal, ["sentimento", "score"]] = ia[["sentimento", "score"]]
            extra.loc[sem_sinal, "confianca"] = ia["confianca"]
            extra.loc[sem_sinal, "fonte"] = "IA"
    saida = trabalho.join(extra).rename(columns={coluna: "comentario"})
    if coluna_nota:
        saida["nota"] = pd.to_numeric(saida[coluna_nota], errors="coerce")
    saida = saida.reset_index(drop=True)
    return saida, avisos


def encurtar(texto: str, limite: int = 160) -> str:
    limpo = " ".join(str(texto).split())
    if len(limpo) <= limite:
        return limpo
    corte = limpo[:limite].rsplit(" ", 1)[0]
    return f"{corte}…"


def ranking(saida: pd.DataFrame, campo: str) -> list[dict]:
    if saida.empty or campo not in saida.columns:
        return []
    campo_peso = "peso_dor" if campo == "dores" else "peso_elogio"
    registros = []
    for _, linha in saida.iterrows():
        pesos = linha[campo_peso] or {}
        for aspecto in linha[campo]:
            registros.append(
                {
                    "aspecto": aspecto,
                    "comentario": linha["comentario"],
                    "peso": float(pesos.get(aspecto, 0.0)),
                }
            )
    if not registros:
        return []
    linhas = pd.DataFrame(registros)
    ordenadas = linhas.sort_values("peso", ascending=(campo == "dores"))
    cita = ordenadas.groupby("aspecto", as_index=False).first()
    resumo = linhas.groupby("aspecto", as_index=False).agg(
        mencoes=("aspecto", "size"),
        gravidade=("peso", "sum"),
    )
    junto = cita.merge(resumo, on="aspecto")
    junto = junto.sort_values(
        ["mencoes", "gravidade"],
        ascending=[False, campo == "dores"],
    )
    return [
        {
            "aspecto": linha["aspecto"],
            "mencoes": int(linha["mencoes"]),
            "citacao": encurtar(linha["comentario"]),
        }
        for _, linha in junto.iterrows()
    ]


def frase_contagem(quantidade: int) -> str:
    if quantidade == 1:
        return "1 comentário"
    return f"{quantidade} comentários"


def concordar(quantidade: int, singular: str, plural: str) -> str:
    if quantidade == 1:
        return f"1 {singular}"
    return f"{quantidade} {plural}"


def porcentagem(parte: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{round(100 * parte / total)}%"


def montar_resumo(saida: pd.DataFrame, dores: list[dict], elogios: list[dict]) -> dict:
    total = len(saida)
    contagem = saida["sentimento"].value_counts() if total else pd.Series(dtype=int)
    positivos = int(contagem.get("Positivo", 0))
    neutros = int(contagem.get("Neutro", 0))
    negativos = int(contagem.get("Negativo", 0))

    frases = [
        f"Li {frase_contagem(total)}.",
        (
            f"{concordar(positivos, 'positivo', 'positivos')} ({porcentagem(positivos, total)}), "
            f"{concordar(neutros, 'neutro', 'neutros')} ({porcentagem(neutros, total)}) e "
            f"{concordar(negativos, 'negativo', 'negativos')} ({porcentagem(negativos, total)})."
        ),
    ]
    if "nota" in saida.columns and saida["nota"].notna().any():
        media = float(saida["nota"].mean())
        frases.append(f"A nota média da planilha é {media:.1f}.")

    recorrentes = [item for item in dores if item["mencoes"] >= MIN_REPETICAO]
    pontuais = [item for item in dores if item["mencoes"] < MIN_REPETICAO]
    if recorrentes:
        nomes = ", ".join(item["aspecto"].lower() for item in recorrentes[:3])
        frases.append(f"As reclamações que mais se repetem: {nomes}.")
        if pontuais:
            soltas = ", ".join(item["aspecto"].lower() for item in pontuais[:3])
            frases.append(f"Apareceram uma vez só: {soltas}.")
    elif pontuais:
        soltas = ", ".join(item["aspecto"].lower() for item in pontuais[:3])
        frases.append(f"Nenhuma reclamação se repetiu. Apareceram uma vez só: {soltas}.")
    elif negativos == 0:
        frases.append("Não há reclamação recorrente nesta leva.")
    else:
        frases.append(
            "Há comentários negativos, mas eles não apontam para o mesmo problema. Vale ler um a um na tabela."
        )

    dores_por_aspecto = {item["aspecto"]: item["mencoes"] for item in dores}
    estaveis = [
        item
        for item in elogios
        if item["mencoes"] > dores_por_aspecto.get(item["aspecto"], 0)
    ]
    if estaveis:
        manter = ", ".join(item["aspecto"].lower() for item in estaveis[:2])
        if dores:
            frases.append(f"O que já funciona: {manter}. Mantenha isso enquanto corrige o resto.")
        else:
            frases.append(f"O que já funciona: {manter}.")

    acoes = []
    for item in recorrentes[:3]:
        texto = ACOES.get(item["aspecto"])
        if texto:
            acoes.append((item["aspecto"], texto))
    # Higiene pede ação mesmo com uma única menção.
    if any(item["aspecto"] == "Higiene" for item in pontuais):
        acoes.append(("Higiene", ACOES["Higiene"]))
    if not acoes:
        if negativos == 0:
            acoes.append(
                (
                    "Manter o padrão",
                    "Este turno não pede mudança de cardápio. Repita a leitura na próxima leva antes de alterar receita ou prazo.",
                )
            )
        else:
            acoes.append(
                (
                    "Ler os casos soltos",
                    "Abra os comentários negativos na tabela e marque em qual etapa o pedido quebrou: cozinha, balcão ou entrega.",
                )
            )

    return {"paragrafo": " ".join(frases), "acoes": acoes}


def grafico_sentimentos(saida: pd.DataFrame) -> go.Figure:
    ordem = ["Positivo", "Neutro", "Negativo"]
    contagem = saida["sentimento"].value_counts().reindex(ordem, fill_value=0)
    figura = go.Figure(
        go.Bar(
            x=list(contagem.index),
            y=[int(valor) for valor in contagem.values],
            marker_color=["#2F6B45", "#8A8175", "#D85A30"],
            text=[int(valor) for valor in contagem.values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    figura.update_layout(
        margin=dict(l=8, r=8, t=24, b=8),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, Segoe UI, sans-serif", color="#1A2420", size=14),
        yaxis=dict(title="Comentários", gridcolor="#D5DAD4", rangemode="tozero"),
        xaxis=dict(title=None),
        bargap=0.35,
    )
    return figura


def grafico_dores(dores: list[dict]) -> go.Figure:
    nomes = [item["aspecto"] for item in dores][::-1]
    valores = [item["mencoes"] for item in dores][::-1]
    figura = go.Figure(
        go.Bar(
            x=valores,
            y=nomes,
            orientation="h",
            marker_color="#D85A30",
            text=valores,
            textposition="outside",
            cliponaxis=False,
        )
    )
    figura.update_layout(
        margin=dict(l=8, r=24, t=24, b=8),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, Segoe UI, sans-serif", color="#1A2420", size=14),
        xaxis=dict(title="Comentários", gridcolor="#D5DAD4", rangemode="tozero"),
        yaxis=dict(title=None),
        bargap=0.35,
    )
    return figura


def faixa_sentimento(saida: pd.DataFrame) -> str:
    ordem = [("Positivo", "pos"), ("Neutro", "neu"), ("Negativo", "neg")]
    total = len(saida)
    contagem = saida["sentimento"].value_counts()
    pedacos = []
    legendas = []
    for nome, classe in ordem:
        quantidade = int(contagem.get(nome, 0))
        fatia = 0 if total == 0 else 100 * quantidade / total
        rotulo = f"{nome} {round(fatia)}%" if fatia >= 16 else ""
        pedacos.append(f'<div class="seg {classe}" style="width:{fatia:.2f}%">{rotulo}</div>')
        legendas.append(f'<span class="leg {classe}">{nome}: {quantidade}</span>')
    return (
        f'<div class="mix" role="img" aria-label="Distribuição dos sentimentos">'
        f'{"".join(pedacos)}</div><div class="legs">{"".join(legendas)}</div>'
    )


def lista_dores(dores: list[dict]) -> str:
    if not dores:
        return "<p class='vazio'>Nenhuma reclamação se repetiu com um motivo claro.</p>"
    itens = []
    for item in dores:
        itens.append(
            "<li>"
            f"<div class='dor-top'><span>{escape(item['aspecto'])}</span>"
            f"<span>{frase_contagem(item['mencoes'])}</span></div>"
            f"<p class='quote'>“{escape(item['citacao'])}”</p>"
            "</li>"
        )
    return f"<ol class='dores'>{''.join(itens)}</ol>"


def bloco_resumo(resumo: dict) -> str:
    acoes = "".join(f"<li><strong>{titulo}.</strong> {texto}</li>" for titulo, texto in resumo["acoes"])
    return (
        "<section class='resumo'>"
        "<h2>O que mudar antes do próximo serviço</h2>"
        f"<p>{resumo['paragrafo']}</p>"
        f"<ul>{acoes}</ul>"
        "</section>"
    )


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,560;9..144,640&family=Outfit:wght@400;500;600&display=swap');

.stApp {
  background: #E7EBE6;
  color: #1A2420;
}
/* Os ícones do Streamlit são spans com fonte de ícones (a palavra "upload" vira o
   desenho). Trocar a fonte deles faz o nome aparecer escrito, por isso ficam de fora. */
.stApp, .stApp p, .stApp li, .stApp label, .stApp textarea, .stApp input,
.stApp span:not([data-testid="stIconMaterial"]) {
  font-family: Outfit, "Segoe UI", sans-serif;
}
.block-container {
  max-width: 1080px;
  padding-top: 1.4rem;
  padding-bottom: 3rem;
}
.board {
  background: #1E3329;
  padding: 2.1rem 2rem 1.9rem;
  margin: 0 0 1.4rem;
}
.board h1 {
  font-family: Fraunces, Georgia, serif;
  font-weight: 560;
  font-size: 3.1rem;
  line-height: 1.02;
  letter-spacing: -0.03em;
  color: #F3EFE4;
  margin: 0 0 0.75rem;
  max-width: 16ch;
}
.board p {
  color: #E4E0D2;
  max-width: 40rem;
  font-size: 1.08rem;
  line-height: 1.45;
  margin: 0;
}
.stApp h2, .stApp h3 {
  font-family: Fraunces, Georgia, serif;
  font-weight: 560;
  letter-spacing: -0.02em;
  color: #1A2420;
}
.resumo {
  margin: 0.4rem 0 1.2rem;
}
.resumo h2 {
  font-family: Fraunces, Georgia, serif;
  font-size: 1.8rem;
  margin: 0 0 0.6rem;
}
.resumo p, .resumo li {
  font-size: 1.05rem;
  line-height: 1.5;
  max-width: 68ch;
}
.resumo ul {
  padding-left: 1.1rem;
  margin: 0.8rem 0 0;
}
.resumo li {
  margin: 0.45rem 0;
}
.mix {
  display: flex;
  width: 100%;
  min-height: 2.6rem;
  background: #D5DAD4;
  overflow: hidden;
  margin-top: 0.4rem;
}
.seg {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #F7F8F6;
  font-weight: 600;
  font-size: 0.92rem;
  min-height: 2.6rem;
  white-space: nowrap;
}
.seg.pos { background: #2F6B45; }
.seg.neu { background: #8A8175; }
.seg.neg { background: #D85A30; }
.legs {
  display: flex;
  gap: 1.2rem;
  margin: 0.45rem 0 0.2rem;
  font-size: 0.92rem;
}
.leg.pos { color: #2F6B45; }
.leg.neu { color: #6A6458; }
.leg.neg { color: #D85A30; }
.dores {
  list-style: none;
  padding: 0;
  margin: 0.4rem 0 0;
}
.dores li {
  padding: 0.75rem 0 0.75rem 0.9rem;
  border-left: 3px solid #D85A30;
  margin-bottom: 0.7rem;
}
.dor-top {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  font-weight: 600;
}
.quote {
  margin: 0.25rem 0 0;
  color: #3C463F;
  max-width: 68ch;
}
.vazio {
  color: #3C463F;
}
div[data-testid="stFileUploader"],
div[data-baseweb="textarea"] {
  background: #F7F8F6;
}
.stApp button[kind="primary"] {
  background-color: #1E3329 !important;
  color: #F3EFE4 !important;
  border: 1px solid #1E3329 !important;
  border-radius: 2px !important;
  font-weight: 600 !important;
}
.stApp button[kind="secondary"] {
  background-color: transparent !important;
  color: #1E3329 !important;
  border: 1px solid #1E3329 !important;
  border-radius: 2px !important;
}
.stApp button:focus-visible {
  outline: 2px solid #D85A30 !important;
  outline-offset: 2px !important;
}
@media (max-width: 700px) {
  .board h1 { font-size: 2.3rem; }
  .legs { flex-direction: column; gap: 0.2rem; }
}
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; animation: none !important; }
}
</style>
"""


def montar_resultado(
    tabela: pd.DataFrame,
    coluna: str,
    coluna_nota: str | None,
    modelo=None,
    ao_progredir: Callable[[float], None] | None = None,
) -> dict:
    saida, avisos = analisar_tabela(tabela, coluna, coluna_nota, modelo, ao_progredir)
    if saida.empty:
        raise ValueError("Não há comentários para ler. A coluna escolhida está vazia.")
    dores = ranking(saida, "dores")
    elogios = ranking(saida, "elogios")
    return {
        "saida": saida,
        "dores": dores,
        "avisos": avisos,
        "motor": "modelo" if modelo is not None else "regras",
        "resumo": montar_resumo(saida, dores, elogios),
    }


def analisar(tabela: pd.DataFrame, coluna: str, coluna_nota: str | None, usar_ia: bool) -> dict:
    """Carrega o modelo (se pedido), mostra progresso e cai para as regras se der errado."""
    modelo = None
    if usar_ia:
        modelo, motivo = carregar_modelo()
        if modelo is None:
            st.warning(f"{motivo} Usei as regras de palavras desta vez.")
    if modelo is None:
        return montar_resultado(tabela, coluna, coluna_nota)
    barra = st.progress(0.0, text="Lendo os comentários com o modelo…")
    try:
        return montar_resultado(
            tabela,
            coluna,
            coluna_nota,
            modelo,
            lambda parte: barra.progress(parte, text=f"Lendo os comentários com o modelo… {round(parte * 100)}%"),
        )
    finally:
        barra.empty()


def render_resultado(resultado: dict) -> None:
    for aviso in resultado["avisos"]:
        st.warning(aviso)
    saida = resultado["saida"]
    st.markdown(faixa_sentimento(saida), unsafe_allow_html=True)
    st.markdown(bloco_resumo(resultado["resumo"]), unsafe_allow_html=True)

    coluna_grafico, coluna_dor = st.columns(2, gap="large")
    with coluna_grafico:
        st.subheader("Distribuição dos sentimentos")
        st.plotly_chart(grafico_sentimentos(saida), use_container_width=True, config={"displayModeBar": False})
    with coluna_dor:
        st.subheader("Reclamações por tema")
        if resultado["dores"]:
            st.plotly_chart(
                grafico_dores(resultado["dores"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.markdown("<p class='vazio'>Nenhuma reclamação com tema claro.</p>", unsafe_allow_html=True)

    st.subheader("Com a frase do cliente")
    st.markdown(lista_dores(resultado["dores"]), unsafe_allow_html=True)

    com_modelo = resultado.get("motor") == "modelo"
    campos = ["comentario", "sentimento", "score", "fonte"] + (["confianca"] if "confianca" in saida.columns else [])
    visao = saida[campos].rename(
        columns={
            "comentario": "Comentário",
            "sentimento": "Sentimento",
            "score": "Score",
            "fonte": "Fonte",
            "confianca": "Confiança",
        }
    )
    ajuda_score = (
        "Regras: abaixo de -0,75 é negativo, acima de 0,75 é positivo. "
        "IA: probabilidade de positivo menos a de negativo, de -1 a 1."
    )
    if com_modelo:
        por_ia = int((saida["fonte"] == "IA").sum())
        st.caption(
            f"As regras de palavras leram {len(saida) - por_ia} comentários. O modelo de IA local "
            f"({MODELO_ID}) leu {por_ia}, os que não tinham nenhuma expressão conhecida."
        )
    else:
        st.caption("Sentimento pelas regras de palavras (modelo de IA desligado ou indisponível).")
    st.subheader("Comentário por comentário")
    st.dataframe(
        visao,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Comentário": st.column_config.TextColumn("Comentário", width="large"),
            "Sentimento": st.column_config.TextColumn("Sentimento", width="small"),
            "Score": st.column_config.NumberColumn("Score", format="%.2f", help=ajuda_score),
            "Fonte": st.column_config.TextColumn("Fonte", width="small", help="Quem decidiu o sentimento."),
            "Confiança": st.column_config.ProgressColumn(
                "Confiança", min_value=0.0, max_value=1.0, format="percent",
                help="Probabilidade que o modelo deu ao sentimento escolhido.",
            ),
        },
    )


def main() -> None:
    st.set_page_config(page_title="Feedback do serviço", page_icon="🍽", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <header class="board">
          <h1>O que os clientes disseram</h1>
          <p>Junte os comentários do salão ou do delivery. A leitura separa o tom, aponta a reclamação que se repete e diz o que ajustar na operação.</p>
        </header>
        """,
        unsafe_allow_html=True,
    )

    if "texto" not in st.session_state:
        st.session_state.texto = ""

    coluna_entrada, coluna_guia = st.columns([1.4, 0.8], gap="large")
    with coluna_entrada:
        arquivo = st.file_uploader("Planilha de comentários", type=["csv", "xlsx"])
        tabela = None
        coluna = None
        nota = None
        if arquivo is not None:
            try:
                tabela = ler_arquivo(arquivo)
            except Exception as exc:
                st.error(str(exc) if str(exc) else "Não consegui abrir esse arquivo. Use .csv ou .xlsx.")
            else:
                if tabela.empty or len(tabela.columns) == 0:
                    st.error("A planilha chegou sem linhas. Inclua uma coluna com os comentários.")
                    tabela = None
                else:
                    sugestao = sugerir_coluna(tabela)
                    indice = list(tabela.columns).index(sugestao)
                    coluna = st.selectbox("Coluna dos comentários", list(tabela.columns), index=indice)
                    nota = coluna_de_nota(tabela, coluna)
                    st.caption("Prévia das primeiras linhas.")
                    st.dataframe(tabela.head(5), use_container_width=True, hide_index=True)
        st.text_area(
            "Ou cole os comentários, um por linha",
            height=220,
            key="texto",
            placeholder="A pizza chegou quente.\nDemorou uma hora e o refrigerante veio errado.",
        )
        if arquivo is not None and tabela is not None:
            st.caption("Com uma planilha enviada, Analisar com IA lê o arquivo e deixa o texto colado de lado.")
    with coluna_guia:
        st.markdown(
            """
            A planilha precisa de uma coluna com o texto. Nomes como comentário, review ou feedback são reconhecidos sozinhos.

            Sem planilha, cole um comentário por linha.

            Cada texto sai como positivo, neutro ou negativo. O resumo destaca o problema que aparece mais de uma vez e o transforma em ação para cozinha, balcão ou entrega. Higiene vira ação já na primeira menção.
            """
        )
        # O callback roda antes da caixa de texto ser criada no próximo ciclo,
        # que é o único momento em que o Streamlit deixa alterar a chave "texto".
        st.button("Preencher com um turno de exemplo", type="secondary", on_click=preencher_exemplo)

    # Identifica o que está na tela agora, para saber se o resultado guardado ficou velho.
    if tabela is not None and coluna is not None:
        fonte_atual = ("arquivo", arquivo.name, arquivo.size, coluna)
    else:
        fonte_atual = ("texto", st.session_state.texto.strip())

    usar_ia = st.toggle(
        "Usar o modelo de IA nos comentários que as regras não reconhecem",
        value=True,
        help="Desligado, esses comentários saem como Neutro.",
    )
    st.caption(
        "O modelo roda neste computador. Ele é baixado da internet só na primeira vez; "
        "os comentários não saem da sua máquina."
    )
    if st.button("Analisar com IA", type="primary", use_container_width=True):
        try:
            if fonte_atual[0] == "arquivo":
                st.session_state.resultado = analisar(tabela, coluna, nota, usar_ia)
                st.session_state.fonte = fonte_atual
            else:
                textos = comentarios_colados(st.session_state.texto)
                if not textos:
                    st.error("Não há comentários para ler. Envie uma planilha ou cole pelo menos uma linha.")
                else:
                    colados = pd.DataFrame({"comentario": textos})
                    st.session_state.resultado = analisar(colados, "comentario", None, usar_ia)
                    st.session_state.fonte = fonte_atual
        except ValueError as exc:
            st.error(str(exc))

    resultado = st.session_state.get("resultado")
    if resultado:
        fonte = st.session_state.get("fonte")
        if fonte != fonte_atual:
            de_onde = "de outra planilha ou coluna" if fonte and fonte[0] == "arquivo" else "de outro texto colado"
            st.warning(
                f"O resultado abaixo é {de_onde}, não do que está na tela agora. "
                "Clique em Analisar com IA para atualizar."
            )
        render_resultado(resultado)


def preencher_exemplo() -> None:
    st.session_state.texto = EXEMPLO.strip()
    st.session_state.resultado = None


if __name__ == "__main__":
    main()
