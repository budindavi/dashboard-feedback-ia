# Dashboard de Análise de Feedback de Clientes com IA

Aplicação web em **Python + Streamlit** que lê comentários de clientes de restaurante ou delivery, classifica o sentimento (Positivo, Neutro, Negativo), aponta as reclamações que se repetem e gera um resumo com ações para cozinha, balcão e entrega.

Tudo roda **na sua máquina**: os comentários não são enviados para nenhum serviço externo.

## Funcionalidades

- **Entrada flexível:** upload de planilha `.csv` ou `.xlsx`, ou texto colado (um comentário por linha).
- **Detecção automática de colunas:** reconhece a coluna de texto (`comentario`, `review`, `feedback`...) e a de nota (`nota`, `rating`, `estrelas`...).
- **Sentimento híbrido (regras + IA local):**
  - um léxico em português trata negação ("não recomendo", "não foi ruim"), intensificadores ("muito", "pouco"), contraste ("mas", "fora isso") e palavras ambíguas ("barata" como inseto ou preço);
  - comentários sem nenhuma expressão conhecida vão para um modelo de linguagem local ([`lxyuan/distilbert-base-multilingual-cased-sentiments-student`](https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student), Apache 2.0).
- **Pontos de dor por tema:** tempo de entrega, temperatura, comida, pedido, embalagem, atendimento, preço, higiene e bebida, cada um com a frase do cliente.
- **Resumo executivo:** distribuição de sentimentos, nota média, reclamações recorrentes e ações sugeridas. Higiene gera ação já na primeira menção.
- **Gráficos** (Plotly) da distribuição de sentimentos e das reclamações por tema.
- **Tabela comentário por comentário**, com score, fonte da decisão (Regras ou IA) e confiança do modelo.

## Requisitos

- Python 3.10 ou mais recente (testado no 3.15)
- Cerca de 1 GB livre: ~250 MB do PyTorch e ~540 MB do modelo, baixado na primeira análise

## Como rodar (Windows / PowerShell)

```powershell
git clone https://github.com/<seu-usuario>/dashboard-feedback-ia.git
cd dashboard-feedback-ia

py -3 -m venv .venv
.venv\Scripts\activate

# Só no Python 3.15 sem compilador C++: evita erro ao instalar o PyYAML
$env:PYYAML_FORCE_LIBYAML = "0"

pip install -r requirements.txt
streamlit run app.py
```

No Linux ou macOS, troque a ativação por `source .venv/bin/activate`.

O app abre em `http://localhost:8501`. Na **primeira análise**, o modelo é baixado do Hugging Face e leva alguns minutos; depois fica em cache e funciona sem internet.

> Sem `torch` e `transformers` o app continua funcionando: avisa e usa só as regras.

## Testando com dados de exemplo

O repositório traz o [`exemplo_reviews.csv`](exemplo_reviews.csv), com 36 pedidos fictícios (colunas `data`, `pedido`, `canal`, `nota`, `comentario`, separador `;`). Arraste o arquivo para o campo de upload e clique em **Analisar com IA**.

Resultado esperado: 35 comentários analisados (uma linha vazia é ignorada) e as reclamações recorrentes **tempo de entrega**, **temperatura** e **comida**.

Também há o botão **Preencher com um turno de exemplo**, que cola 12 comentários na caixa de texto.

## Formato da planilha

| Coluna | Obrigatória | Observação |
|---|---|---|
| Texto do comentário | Sim | Detectada pelo nome ou pela coluna com textos mais longos. Dá para trocar na tela. |
| Nota | Não | Numérica. Usada para mostrar a nota média. |

Encoding UTF-8, Windows-1252 e Latin-1 são aceitos; o separador (`,` ou `;`) é detectado sozinho. O limite é de 5.000 comentários por análise.

## Como funciona

```
comentário ─► regras de palavras ──► achou expressão? ── sim ─► sentimento pelas regras
                    │                       └─ não ─► modelo de IA local ─► sentimento
                    └─► temas (dores e elogios) ─► ranking ─► resumo e ações
```

Os modelos leves que testamos erravam mais que as regras em frases curtas de restaurante (um deles classificou "Péssimo" como positivo). Por isso o modelo só decide quando as regras não reconhecem nada, como em "o app travou e o cupom não funcionou" ou "chegou às 22h. Parabéns, hein".

## Limitações conhecidas

- Os temas e o resumo vêm de regras e textos prontos, não de um modelo generativo.
- Reclamações fora dos 9 temas (app, cupom, cancelamento) entram no sentimento, mas não viram tema.
- Ironia e gírias só são captadas quando caem no modelo.
- O vocabulário cobre restaurante e delivery em português do Brasil.

## Estrutura

```
app.py               # aplicação completa (regras, modelo, gráficos e interface)
requirements.txt     # dependências
exemplo_reviews.csv  # planilha de teste
```

## Tecnologias

[Streamlit](https://streamlit.io) · [pandas](https://pandas.pydata.org) · [Plotly](https://plotly.com/python/) · [Hugging Face Transformers](https://huggingface.co/docs/transformers) · [PyTorch](https://pytorch.org)
