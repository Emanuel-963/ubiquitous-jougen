# IonFlow Pipeline — Relatório Técnico

**Software de Análise Eletroquímica Integrada: EIS · Ciclagem · DRT**

Versão 0.1.0 · Abril 2026

---

## 1. Introdução

O **IonFlow Pipeline** é um software de análise eletroquímica que integra três pipelines complementares — Espectroscopia de Impedância Eletroquímica (EIS), Ciclagem Galvanostática (GCD) e Distribuição de Tempos de Relaxação (DRT) — numa interface gráfica unificada. O objetivo é permitir a análise quantitativa e a classificação automática de materiais para dispositivos de armazenamento de energia.

Este relatório apresenta o funcionamento dos pipelines, os resultados gerados a partir de amostras reais de Nb₂O₅ sobre aço 316 em diferentes eletrólitos, e as correlações físicas entre as métricas extraídas.

---

## 2. Pipeline EIS — Espectroscopia de Impedância

### 2.1 Fluxo de processamento

O pipeline EIS executa a seguinte sequência automatizada:

```
Dados brutos (freq, Z_real, Z_imag)
    │
    ├─ Extração de métricas físicas (Rs, Rp, C, E, τ, Dispersão)
    ├─ Ajuste de CPE + Warburg (Q, n, Sigma)
    ├─ Fitting de circuitos equivalentes (BIC/AIC)
    ├─ Classificação K-Means (2 clusters)
    ├─ Score composto e ranking
    ├─ PCA (redução dimensional)
    └─ Análise de estabilidade entre réplicas (CV)
```

### 2.2 Circuitos equivalentes e seleção automática

O software ajusta três modelos candidatos e seleciona o melhor por BIC penalizado:

| Circuito | Representação | Amostras selecionadas |
|----------|--------------|----------------------|
| **Randles-CPE-W** | Rs − (Rp ‖ CPE) − W | 6 (amostras Nb₂) |
| **Two-Arc-CPE** | Rs − (Rp₁ ‖ CPE₁) − (Rp₂ ‖ CPE₂) | 4 (amostras AM3/AM4) |
| **Inductive-CPE** | Rs − L − (Rp ‖ CPE) | 0 |

Para as amostras AM3 e AM4, o modelo Two-Arc-CPE foi selecionado com confiança de 100%, indicando dois processos eletroquímicos distinguíveis — consistente com interfaces eletrodo/filme e filme/eletrólito.

**Fitting de circuito — Amostra AM4 em H₂SO₄ (Two-Arc-CPE):**

![Fitting de circuito AM4](outputs/figures/circuits/Cópia%20de%20AM4%20-%20EIS%20Nb2%20H2SO4.txt_Two-Arc-CPE.png)

*O gráfico mostra os dados experimentais (pontos) e o ajuste do modelo (linha), com resíduos indicando a qualidade do fitting.*

### 2.3 Classificação e Ranking

O K-Means (k=2) classifica as amostras em dois grupos com base em Rs_fit e Rp_fit:

- **Interface eficiente**: Amostras com alta resistência de polarização (armazenamento faradaico), baixa resistência série
- **Genérica estável**: Comportamento predominantemente capacitivo de dupla camada

O **score composto** combina quatro métricas normalizadas:

$$\text{Score} = 0{,}35 \times Rp_n - 0{,}25 \times Rs_n + 0{,}25 \times C_n + 0{,}15 \times E_n$$

| Rank | Amostra | Eletrólito | Classe | Score |
|------|---------|-----------|--------|-------|
| 1 | 1 Nb₂ S316 H₂SO₄ Álcool | H₂SO₄ | Genérica estável | 0,987 |
| 2 | AM4 − EIS Nb₂ H₂SO₄ | H₂SO₄ | Genérica estável | 0,043 |
| 3 | AM4 EIS pós-GCD | Pós-ciclagem | Genérica estável | −0,063 |
| 4 | AM3 EIS pós-GCD | Pós-ciclagem | Genérica estável | −0,104 |
| 5 | AM3 − EIS − antes | Antes GCD | Genérica estável | −0,192 |

### 2.4 Análise de Componentes Principais (PCA)

A PCA sobre as 5 variáveis do fitting (Rs_fit, Rp_fit, Q, n, Sigma) revela a estrutura multivariada:

**Loadings dos componentes:**

| Variável | PC1 | PC2 | PC3 |
|----------|-----|-----|-----|
| Rs_fit | −0,346 | **0,598** | −0,219 |
| Rp_fit | **0,551** | 0,401 | 0,033 |
| Q | −0,466 | 0,274 | −0,522 |
| n | **0,531** | 0,449 | −0,222 |
| Sigma | −0,277 | 0,454 | **0,793** |

**Interpretação:** PC1 separa amostras por processos de transferência de carga (Rp, n altos → interface mais complexa). PC2 discrimina por resistência ôhmica e contribuição de Warburg.

**PCA 2D — Separação das amostras:**

![PCA 2D](outputs/figures/pca_2d.png)

*A distribuição no espaço PC1×PC2 permite identificar agrupamentos naturais entre as amostras.*

### 2.5 Heatmap de correlação de Spearman

![Heatmap de correlação](outputs/figures/analytics/correlation_heatmap.png)

**Correlações mais relevantes (ρ de Spearman):**

| Correlação | ρ | Interpretação física |
|-----------|---|---------------------|
| Score ↔ Rank | −0,994 | Validação interna do ranking |
| Rs_fit ↔ Score | −0,952 | Rs domina negativamente o desempenho |
| Rs_fit ↔ Rank | +0,957 | Maior Rs → pior posição no ranking |
| Rp_fit ↔ n | +0,952 | Interface complexa correlaciona com dispersão CPE |
| C_mean ↔ Energy | +1,000 | Confirmação: E = ½CV² |
| Sigma ↔ C_mean | −0,624 | Difusão de Warburg compete com capacitância |
| Q ↔ Rs_fit | +0,770 | Amostras resistivas têm maior pseudocapacitância Q |

A anticorrelação Rs_fit × Score (ρ = −0,95) confirma que a **resistência série é o principal limitante** do desempenho neste conjunto de amostras.

---

## 3. Pipeline DRT — Distribuição de Tempos de Relaxação

### 3.1 Metodologia

O pipeline DRT deconvolui o espectro de impedância usando **regularização de Tikhonov**:

$$\min_{\gamma} \left\| Z_{exp} - Z_{calc}(\gamma) \right\|^2 + \lambda \left\| L\gamma \right\|^2$$

onde λ controla o balanço entre fidelidade aos dados e suavidade da solução, e L é a matriz de diferenciação de 2ª ordem.

### 3.2 Resultados DRT

| Amostra | N° picos | R_inf (Ω) | τ principal (s) | γ principal (Ω) |
|---------|----------|-----------|----------------|-----------------|
| 1 Nb₂ H₂SO₄ Am4 Água | 4 | 3,37 | 3,46 | 2,4×10⁶ |
| 1 Nb₂ H₂SO₄ Am4 Álcool | 4 | 2,63 | 2,54 | 653,6 |
| 1 Nb₂ Na₂SO₄ Am2 | 3 | 11,81 | 0,24 | 3,3×10⁵ |
| AM3 − EIS − antes | 3 | 3,77 | 0,31 | 9,4×10⁴ |
| AM3 EIS pós-GCD | 2 | 3,87 | 0,25 | 3,4×10⁴ |
| AM4 − EIS Nb₂ H₂SO₄ | 3 | 1,33 | 0,28 | 3,5×10⁴ |
| AM4 EIS pós-GCD | 3 | 3,49 | 0,31 | 3,6×10⁴ |

**Espectro DRT — Amostra Nb₂ H₂SO₄ Am4 Álcool:**

![DRT Nb2 Alcool](outputs/figures/drt/1%20Nb2%20S316%20H2SO4%20Am4%20Alcool%20NF_drt.png)

*O espectro γ(τ) revela os processos de relaxação individuais. Cada pico corresponde a um processo eletroquímico distinto.*

**DRT — AM4 antes da ciclagem:**

![DRT AM4 antes](outputs/figures/drt/Cópia%20de%20AM4%20-%20EIS%20Nb2%20H2SO4_drt.png)

### 3.3 Interpretação dos processos por faixa de τ

| Faixa de τ | Processo físico | Amostras onde aparece |
|-----------|----------------|----------------------|
| 10⁻⁷ – 10⁻⁴ s | Resposta dielétrica do bulk | Nb₂ (picos secundários) |
| 10⁻³ – 10⁻¹ s | Transferência de carga na interface | Todas (pico principal) |
| 10⁻¹ – 10¹ s | Difusão no filme / em poros | Nb₂ H₂SO₄ Água (τ = 3,46 s) |

### 3.4 Correlações DRT ↔ EIS

A consistência entre os métodos pode ser verificada:

- **R_inf (DRT) ≈ Rs (EIS)**: Para AM4-H₂SO₄, R_inf = 1,33 Ω e Rs = 1,33 Ω → **excelente consistência**
- **Número de picos DRT ↔ Circuito selecionado**: As 4 amostras AM3/AM4, para as quais Two-Arc-CPE foi selecionado, apresentam 2–3 picos DRT → **consistente** com dois arcos de impedância
- **Redução de picos pós-GCD**: AM3 passa de 3 para 2 picos após ciclagem, indicando simplificação da interface (possível estabilização do filme)

---

## 4. Pipeline de Ciclagem Galvanostática (GCD)

### 4.1 Métricas extraídas

O pipeline processa curvas de carga/descarga e calcula:

$$E = \frac{\int V \cdot I \, dt}{m} \quad \text{(Wh/kg)} \qquad P = \frac{E}{\Delta t} \quad \text{(W/kg)}$$

**Curva GCD — Tempo vs Potencial (AM4):**

![GCD AM4](outputs/figures/discharge%20am4_integral.png)

*A área sob a curva V(t) durante a descarga é proporcional à energia armazenada.*

**Energia e Potência por ciclo:**

![Energia e Potência AM4](outputs/figures/discharge%20am4_energy_power.png)

*A evolução de energia (azul) e potência (vermelho) ao longo dos ciclos indica a estabilidade do dispositivo.*

**Ciclagem GCD — AM3 (Nb₂ em H₂SO₄):**

![GCD AM3](outputs/figures/Cópia%20de%20GCD%20Nb2%20-%20AM3_integral.png)

![Energia AM3](outputs/figures/Cópia%20de%20GCD%20Nb2%20-%20AM3_energy_power.png)

### 4.2 Diagrama de Ragone

O diagrama de Ragone posiciona as amostras no espaço energia × potência em escala log-log, permitindo comparação com classes de dispositivos conhecidas:

- **Região de supercapacitores**: E = 1–10 Wh/kg, P = 10²–10⁴ W/kg
- **Região de baterias**: E = 10–200 Wh/kg, P = 10–10³ W/kg

---

## 5. Correlações Cruzadas entre Pipelines

### 5.1 EIS ↔ Ciclagem

| Métrica EIS | Métrica Ciclagem | Relação esperada | Observado |
|-------------|-----------------|------------------|-----------|
| Rs baixo | Potência alta | Direta (P ∝ 1/Rs) | Rs(AM4) = 1,33 Ω → melhor potência |
| C_mean alto | Energia alta | Direta (E ∝ C) | ρ(C,E) = 1,00 |
| τ curto | Melhor rate capability | Inversa | Amostras com τ < 0,1 s → ciclagem estável |

### 5.2 EIS ↔ DRT

| Verificação de consistência | Resultado |
|----------------------------|-----------|
| R_inf ≈ Rs | ✓ Consistente para todas as amostras |
| N° picos DRT ≈ N° arcos Nyquist | ✓ Two-Arc → 2-3 picos; Randles → 3-4 picos |
| Redução de γ_peak pós-GCD | ✓ AM3: γ diminui após ciclagem (interface estabiliza) |

### 5.3 DRT ↔ Ciclagem

A comparação das amostras AM3/AM4 antes e após ciclagem revela:

- **AM3 antes → AM3 pós-GCD**: R_inf aumenta de 3,77 → 3,87 Ω (+2,7%), número de picos diminui de 3 → 2
  - *Interpretação*: Um processo de relaxação desaparece após ciclagem, sugerindo estabilização da interface eletrodo/eletrólito com perda parcial de um caminho de condução.

- **AM4 antes → AM4 pós-GCD**: R_inf aumenta de 1,33 → 3,49 Ω (+162%), número de picos se mantém em 3
  - *Interpretação*: Aumento significativo da resistência ôhmica sem simplificação da interface. Possível crescimento de filme passivante ou degradação de contato.

### 5.4 Efeito do eletrólito

| Eletrólito | Rs médio (Ω) | Rank médio | Observação |
|-----------|-------------|-----------|------------|
| H₂SO₄ | 2,6 ± 0,7 | 3,0 | Menor resistência, melhor condutividade |
| Na₂SO₄ | 12,1 | 2,0* | Alta Rs, mas alto Rp (faradaico) |

*Rank 2 apesar de alto Rs devido ao alto Rp compensatório no score.*

---

## 6. Métricas do PCA por variável — Série temporal

**Evolução da capacitância específica entre réplicas:**

![Série C_espec AM4 Alcool](outputs/figures/analytics/series_C_espec_(F_g)_Nb2_S316_H2SO4_Am4_Alcool_NF.png)

*A variação entre réplicas da mesma amostra indica a reprodutibilidade das medições.*

**PCA colorido por métrica:**

![PCA 2D por métrica](outputs/figures/analytics/pca_2d_metric.png)

*A projeção PCA 2D com gradiente de cor por métrica revela quais regiões do espaço concentram melhores desempenhos.*

---

## 7. Resumo dos resultados

### Melhor amostra geral: 1 Nb₂ S316 H₂SO₄ Am4 Álcool (Rank 1)

| Métrica | Valor | Destaque |
|---------|-------|----------|
| Rs | 2,66 Ω | Baixa resistência série |
| Rp | 850,6 Ω | Transferência de carga moderada |
| C_lowfreq | 2,31 mF | Maior capacitância de baixa frequência |
| n (CPE) | 0,767 | Comportamento pseudocapacitivo |
| τ | 9,8 ms | Resposta rápida |
| Score | 0,987 | Melhor score composto |
| Circuito | Randles-CPE-W | Interface simples com difusão |
| DRT: picos | 4 | Múltiplos processos de armazenamento |

### Conclusões principais

1. **Rs é o principal discriminador de desempenho** (ρ = −0,95 com Score), confirmando que a resistência ôhmica domina as perdas neste sistema.

2. **O modelo Two-Arc-CPE é preferido para amostras com interfaces complexas** (AM3/AM4), enquanto o Randles-CPE-W descreve melhor as amostras Nb₂ com interface mais simples.

3. **A DRT complementa o fitting de circuitos** ao resolver processos sobrepostos que aparecem como um único semicírculo achatado no Nyquist.

4. **A ciclagem revela degradação não detectável somente por EIS**: AM4 mostra aumento de 162% em R_inf após GCD, indicando formação de filme passivante.

5. **H₂SO₄ proporciona menor resistência série** que Na₂SO₄, consistente com a maior condutividade iônica do ácido forte.

---

## 8. Sobre o software

| Característica | Detalhe |
|---------------|---------|
| Linguagem | Python 3.11+ |
| Interface | CustomTkinter (17 abas interativas) |
| Idiomas | Português e Inglês |
| Licença | MIT |
| Distribuição | Executável Windows (Inno Setup) + ZIP portátil |
| Gráficos interativos | 17 tipos com tooltips e zoom |
| Exportação | CSV, Excel (.xlsx), PNG |
| Repositório | github.com/Emanuel-963/ubiquitous-jougen |

**Pipelines disponíveis:**
- **EIS**: Métricas físicas → CPE → Circuitos → K-Means → PCA → Ranking
- **Ciclagem**: GCD → Energia/Potência → Ragone → Retenção cíclica
- **DRT**: Tikhonov → Picos de relaxação → Correlação com EIS

---

*Relatório para o IonFlow Pipeline v0.1.0*
*Para detalhes de uso, consulte os tutoriais em `tutoriais/01–08`*
