# Apresentação: EIS Analytics Framework

## Objetivo breve (1 slide)
- Contexto: análise de EIS para caracterização de materiais eletroquímicos
- Problema: automatizar extração de métricas físicas e facilitar comparação entre amostras
- Contribuição: pipeline reproducível com métricas, PCA e visualizações

## Estrutura da apresentação (5–7 slides)
1. Título + Autoria (1 slide)
   - Título do trabalho, autor(es), orientador, afiliação
2. Motivação & Objetivos (1–2 slides)
   - Por que EIS, aplicações e lacunas atuais
   - Objetivos específicos do framework
3. Abordagem & Métodos (1–2 slides)
   - Pipeline: carregamento, pré-processamento, extração de métricas, PCA, ranking
   - Principais métricas físicas (Rs, Rp, C_eff, Tau, Energy)
4. Resultados (1–2 slides)
   - Exemplo de figura (Nyquist, PCA 2D) e tabela de resumo
   - Interpretação de um caso: como o framework ajuda na decisão
5. Reprodutibilidade & Uso (1 slide)
   - Como rodar, testes, requisitos, e instruções para regenerar figuras
6. Conclusões & Trabalho futuro (1 slide)
   - Limitações, próximos passos e aplicações potenciais
7. Agradecimentos & Perguntas (1 slide)

## Notas para a apresentação (peaker notes)
- 5–7 minutos: foco na motivação, resultado exemplar e reproducibilidade.
- 10–12 minutos: explique com mais detalhe a metodologia e discuta limitações.

## Materiais suplementares
- `scripts/regenerate_figures.py` — script para gerar exemplos de figuras
- Testes unitários no diretório `tests/` comprovam que as funções principais estão cobertas

