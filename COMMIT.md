# Commit Log – EIS Analytics Framework

Este arquivo documenta decisões técnicas, mudanças estruturais e evoluções metodológicas do projeto.

---

## Commit 001 – Estrutura inicial
- Criação da arquitetura de pastas
- Separação modular do código
- Definição do fluxo físico de dados EIS

## Commit 002 – Núcleo físico
- Implementação da capacitância efetiva espectral
- Cálculo de Rs e Rp sem uso de circuito equivalente
- Introdução de energia acumulada e constante de tempo

## Commit 003 – Análise multivariada
- Implementação de PCA
- Padronização das features físicas
- Interpretação física dos componentes principais

## Commit 004 – Classificação de aplicação
- Heurísticas físicas para:
  - Supercapacitor
  - Bateria
  - Célula eletroquímica genérica
  - Fotovoltaica / PEC
- Ranking por aplicação

## Commit 005 – Visualização
- Gráficos de Nyquist
- Diagramas de Bode
- Projeções PCA

---

## Observação
Este projeto prioriza:
- Extração direta de propriedades físicas
- Interpretabilidade dos resultados
- Reprodutibilidade científica
