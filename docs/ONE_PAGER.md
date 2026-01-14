# EIS Analytics Framework — One-page summary

Resumo rápido:
- O EIS Analytics Framework automatiza a análise de Espectroscopia de Impedância Eletroquímica (EIS) para caracterização e comparação de materiais.
- Extrai métricas físicas robustas (Resistência série Rs, Resistência de polarização Rp, Capacitância efetiva dependente de frequência, Energia acumulada, Tau, Dispersion Index) e realiza análise multivariada (PCA) para comparação entre amostras.

Pontos-chave para comunicar:
- Reprodutibilidade: ambiente virtual, testes unitários (pytest) e scripts para regenerar figuras.
- Uso prático: executar `python main.py` com os arquivos EIS em `data/raw/` gera `outputs/` com figuras e tabelas.
- Código: modular, documentado com docstrings e tipos, test coverage atual ~76%.

Como demonstrar (2–3 minutos):
1. Mostrar o repositório e o README breve.
2. Rodar `python scripts/regenerate_figures.py` para gerar uma PCA 2D de exemplo.
3. Mostrar resultados em `outputs/figures` e `outputs/tables`.

Links úteis:
- Código: https://github.com/Emanuel-963/ubiquitous-jougen
- Pull request com melhorias de qualidade: https://github.com/Emanuel-963/ubiquitous-jougen/pull/1

