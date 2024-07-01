# Tennis Tracking

Este repositório contém uma solução para rastreamento de partidas de tênis utilizando tanto visão computacional clássica quanto deep learning.

## Estrutura do Projeto

- `input_videos/`: Contém vídeos de partidas de tênis a serem processados.
- `frames/`: Armazena os frames extraídos dos vídeos.
- `output_videos/`: Guarda os resultados dos vídeos processados com anotações de rastreamento.

## Visão Computacional Clássica

Os scripts no diretório `classic` implementam técnicas tradicionais de visão computacional para rastreamento dos jogadores. As técnicas incluem:
- Detecção de movimento
- Análise de fundo

### Como Usar
1. Navegue até o diretório `classic`.
2. Execute o arquivo `main`

## Deep Learning

Os scripts no diretório `deep` utilizam modelos de deep learning, especificamente YOLOv8, para a detecção e rastreamento de jogadores e bolas.

### Como Usar
1. Navegue até o diretório `deep`.
2. Execute o arquivo `main`

## Procedimento Geral

1. **Pré-processamento:** Converta os vídeos em frames.
2. **Detecção:** Aplique os modelos clássicos e de deep learning para identificar jogadores e bolas.
3. **Pós-processamento:** Gere vídeos de saída com anotações.

## Requisitos

- Python 3.x
- Bibliotecas necessárias:
    ```sh
    pip install -r requirements.txt
    ```
