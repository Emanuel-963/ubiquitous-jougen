# Circuitos Equivalentes — Referência Completa (37 circuitos)

> Documento de referência para os 37 circuitos equivalentes utilizados no
> preditor de circuitos do IonFlow Pipeline.
>
> **Estrutura do sistema**
> - **Todos os 37 circuitos** estão registados em `src/circuit_registry.py` (fitting real)
>   **e** em `scripts/gen_synthetic_eis.py` (dados sintéticos para treino do classificador ML).
> - **11 circuitos base** — núcleo histórico do sistema; incluem os casos de uso mais comuns
>   (Randles, ZARC duplo/triplo, difusão, indutivo, revestimento, EDLC).
> - **4 circuitos especializados** — `Porous-Coating-TLM`, `MXene-Intercalation`,
>   `De-Levie-TLM`, `Pseudo-Capacitance-CPE` — adicionados ao gerador sintético na v0.4.10.
> - **22 circuitos estendidos (EXT-01 a EXT-22)** — adicionados ao registry na v0.4.10;
>   cobrem topologias compostas (indutivo + difusão, TLM + ZARC, Gerischer com arcos múltiplos).
> - **Classificador ML**: RandomForest com `CalibratedClassifierCV(method='sigmoid')`;
>   12 features espectrais incluindo 3 residuais de Kramers-Kronig.

---

## Índice rápido

| # | Nome | Topologia | Nº params | Família principal |
|---|------|-----------|-----------|-------------------|
| 1 | `Randles-CPE-W` | Rs − (Rp‖CPE) − W | 5 | Randles modificado |
| 2 | `Two-Arc-CPE` | Rs − (Rp1‖CPE1) − (Rp2‖CPE2) | 7 | ZARC duplo |
| 3 | `Inductive-CPE` | Rs − L − (Rp‖CPE) | 5 | Indutivo |
| 4 | `Coating-CPE` | Rs − (Rcoat‖CPEcoat) − (Rct‖CPEdl) | 7 | Revestimento |
| 5 | `Warburg-Finite` | Rs − (Rp‖CPE) − W_tanh | 6 | Difusão finita |
| 6 | `ZARC-ZARC-W` | Rs − ZARC₁ − ZARC₂ − W | 8 | Multi-ZARC + difusão |
| 7 | `Simple-RC` | Rs − (Rp‖C) | 3 | Baseline |
| 8 | `CPE-Simple` | Rs − CPE | 3 | EDLC puro |
| 9 | `Warburg-Short` | Rs − (Rp‖CPE) − W_coth | 6 | Difusão refletiva |
| 10 | `Gerischer` | Rs − (Rp‖CPE) − Z_Ger | 6 | Gerischer |
| 11 | `Three-ZARC` | Rs − ZARC₁ − ZARC₂ − ZARC₃ | 10 | Eletrolitos sólidos |
| 12 | `Porous-Coating-TLM` | Rs − (Rcoat‖Cpore) − (Rct‖CPEdl) | 6 | TLM poroso |
| 13 | `MXene-Intercalation` | Rs − (Rsei‖CPEsei) − (Rct‖CPEdl) − W_fin | 9 | MXene |
| 14 | `De-Levie-TLM` | Rs + √(Ri/Ydl)·coth(L√(Ri·Ydl)) | 5 | TLM De Levie |
| 15 | `Pseudo-Capacitance-CPE` | Rs − (Rct‖CPEdl) − (Rads‖Cads) | 6 | Pseudocapacitância |
| 16 | `Rs-ZARC-TLM` | Rs − ZARC − TLM | 7 | EXT-01 |
| 17 | `Rs-ZARC-ZARC-Wfinite` | Rs − ZARC₁ − ZARC₂ − W_fin | 9 | EXT-02 |
| 18 | `Rs-ZARC-ZARC-Wshort` | Rs − ZARC₁ − ZARC₂ − W_coth | 9 | EXT-03 |
| 19 | `Rs-ZARC-ZARC-Gerischer` | Rs − ZARC₁ − ZARC₂ − Ger | 9 | EXT-04 |
| 20 | `Rs-ZARC-ZARC-TLM` | Rs − ZARC₁ − ZARC₂ − TLM | 10 | EXT-05 |
| 21 | `Rs-RC-ZARC-W` | Rs − (R‖C) − ZARC − W | 7 | EXT-06 |
| 22 | `Rs-ZARC-RC-Wfinite` | Rs − ZARC − (R‖C) − W_fin | 8 | EXT-07 |
| 23 | `Rs-L-ZARC-W` | Rs − L − ZARC − W | 6 | EXT-08 |
| 24 | `Rs-L-ZARC-Wfinite` | Rs − L − ZARC − W_fin | 7 | EXT-09 |
| 25 | `Rs-L-ZARC-ZARC` | Rs − L − ZARC₁ − ZARC₂ | 8 | EXT-10 |
| 26 | `Rs-ZARC-ZARC-ZARC-W` | Rs − ZARC₁ − ZARC₂ − ZARC₃ − W | 11 | EXT-11 |
| 27 | `Rs-ZARC-ZARC-ZARC-Wfinite` | Rs − ZARC₁ − ZARC₂ − ZARC₃ − W_fin | 12 | EXT-12 |
| 28 | `Rs-ZARC-CPE` | Rs − ZARC − CPE | 6 | EXT-13 |
| 29 | `Rs-RC-W` | Rs − (R‖C) − W | 4 | EXT-14 |
| 30 | `Rs-RC-Wfinite` | Rs − (R‖C) − W_fin | 5 | EXT-15 |
| 31 | `Rs-ZARC-ZARC-CPE` | Rs − ZARC₁ − ZARC₂ − CPE | 9 | EXT-16 |
| 32 | `Rs-RC-ZARC-Wfinite` | Rs − (R‖C) − ZARC − W_fin | 8 | EXT-17 |
| 33 | `Rs-ZARC-RC-Wshort` | Rs − ZARC − (R‖C) − W_coth | 8 | EXT-18 |
| 34 | `Rs-L-ZARC-ZARC-W` | Rs − L − ZARC₁ − ZARC₂ − W | 9 | EXT-19 |
| 35 | `Rs-TLM` | Rs − TLM puro | 4 | EXT-20 |
| 36 | `Rs-ZARC-ZARC-ZARC-Gerischer` | Rs − ZARC₁ − ZARC₂ − ZARC₃ − Ger | 12 | EXT-21 |
| 37 | `Rs-ZARC-TLM-W` | Rs − ZARC − TLM − W | 8 | EXT-22 |

---

## Notação e elementos primitivos

| Símbolo | Descrição | Equação de impedância |
|---------|-----------|----------------------|
| **Rs** | Resistência em série (eletrólito) | $Z = R_s$ |
| **R** | Resistor puro | $Z = R$ |
| **C** | Capacitor ideal | $Z = \frac{1}{j\omega C}$ |
| **L** | Indutor | $Z = j\omega L$ |
| **CPE** | Constant-Phase Element | $Z = \frac{1}{Q(j\omega)^n}$ |
| **ZARC** | R‖CPE (arco deprimido) | $Z = \frac{1}{1/R + Q(j\omega)^n}$ |
| **W** | Warburg semi-infinito | $Z = \frac{\sigma}{\sqrt{\omega}} (1-j)$ |
| **W_fin** | Warburg finito (tanh) | $Z = \frac{R_d \tanh(\sqrt{j\omega T_d})}{\sqrt{j\omega T_d}}$ |
| **W_coth** | Warburg refletivo (coth) | $Z = \frac{R_d \coth(\sqrt{j\omega T_d})}{\sqrt{j\omega T_d}}$ |
| **Ger** | Gerischer | $Z = \frac{R_g}{\sqrt{1 + j\omega T_g}}$ |
| **TLM** | De Levie transmission-line | $Z = \sqrt{\frac{R_i}{Y_{dl}}} \cdot \coth\!\left(L\sqrt{R_i Y_{dl}}\right)$ |

---

## Circuitos base (11) — presentes no registry e no gerador sintético

---

### 1. `Randles-CPE-W`

**Topologia:** `Rs − (Rp ‖ CPE) − W`

**Diagrama:**
```
Rs ──┬──── W ────
     │
  Rp ‖ CPE
     │
     └───────────
```

**Parâmetros:**

| Parâmetro | Significado físico | Intervalo típico |
|-----------|-------------------|-----------------|
| Rs | Resistência ôhmica do eletrólito (Ω) | 0,05 – 4 Ω |
| Rp | Resistência de transferência de carga (Ω) | 1 – 150 Ω |
| Q | Pseudo-capacitância CPE (F·s^(n−1)) | 2×10⁻⁵ – 5×10⁻³ |
| n | Expoente CPE (1 = capacitor ideal) | 0,72 – 0,97 |
| Sigma | Coeficiente de Warburg (Ω·s^−½) | 0,005 – 3 |

**Sistemas típicos:** baterias de Li-ion, supercapacitores com difusão, corrosão com controle misto cinético-difusivo, eletrodos de células a combustível.

**Características no Nyquist:** semicírculo deprimido a altas frequências + cauda de Warburg (45°) a baixas frequências.

---

### 2. `Two-Arc-CPE`

**Topologia:** `Rs − (Rp1 ‖ CPE1) − (Rp2 ‖ CPE2)`

**Parâmetros:** Rs, Rp1, Q1, n1, Rp2, Q2, n2 (7 parâmetros)

**Significado físico:** dois processos com constantes de tempo distintas — tipicamente SEI + transferência de carga, ou bulk + contorno de grão.

**Sistemas típicos:** SOFC, eletrolitos sólidos (bulk + contorno de grão), metais revestidos com duas interfaces, células Li-ion (SEI + CT).

**Características:** dois semicírculos deprimidos sobrepostos; distinguível do `Three-ZARC` pela ausência de um terceiro arco.

---

### 3. `Inductive-CPE`

**Topologia:** `Rs − L − (Rp ‖ CPE)`

**Parâmetros:** Rs, L, Rp, Q, n (5 parâmetros)

**Significado físico:**
- **L** — indutância em série devida a cabos/conectores ou a intermediários adsorvidos (processo de adsorção em baixas frequências).

**Sistemas típicos:** metais em corrosão com intermediários adsorvidos, sistemas com cabos longos, PEM a altas frequências.

**Características:** loop indutivo no primeiro quadrante a altas frequências (−Im(Z) < 0); os dados do loop devem ser excluídos do fit se L for artefato de cabo.

---

### 4. `Coating-CPE`

**Topologia:** `Rs − (Rcoat ‖ CPEcoat) − (Rct ‖ CPEdl)`

**Parâmetros:** Rs, Rcoat, Qcoat, ncoat, Rct, Qdl, ndl (7 parâmetros)

**Significado físico:**
- **Rcoat / CPEcoat** — resposta capacitiva-resistiva do revestimento (poros de elétrolito + capacitância dielétrica do filme).
- **Rct / CPEdl** — interface metal/eletrólito subjacente (transferência de carga + dupla camada).

**Sistemas típicos:** revestimentos orgânicos sobre aço (proteção contra corrosão), alumínio anodizado, estruturas metálicas pintadas, implantes biomédicos.

**Diferença de `Porous-Coating-TLM`:** aqui ambas as ramificações usam CPE; em `Porous-Coating-TLM` o revestimento usa capacitor ideal (Cpore), capturando melhor a separação física entre capacitância dielétrica e rugosidade da interface.

---

### 5. `Warburg-Finite`

**Topologia:** `Rs − (Rp ‖ CPE) − W_tanh`

**Parâmetros:** Rs, Rp, Q, n, Rd, Td (6 parâmetros)

**Significado físico:**
- **Rd** — resistência de difusão (Ω); proporcional à espessura da camada difusiva.
- **Td** — constante de tempo de difusão (s) = L²/D; L = espessura, D = difusividade.

**Boundary condition:** transmissiva (eletrodo receptor no fundo da camada), produz $Z \to R_d$ a ω → 0.

**Sistemas típicos:** baterias de filme fino, supercapacitores com eletrodo poroso, células simétricas para estudo de eletrólito, GDL de células a combustível.

**Características:** cauda de difusão que se curva para eixo real a baixas frequências (diferente de W_coth que sobe para eixo imaginário).

---

### 6. `ZARC-ZARC-W`

**Topologia:** `Rs − ZARC₁ − ZARC₂ − W`

**Parâmetros:** Rs, R1, Q1, n1, R2, Q2, n2, Sigma (8 parâmetros)

**Sistemas típicos:** eletrodos de bateria com múltiplas camadas (SEI + CT + difusão), cerâmicas mistas (catodos SOFC), corrosão complexa com múltiplas interfaces.

---

### 7. `Simple-RC`

**Topologia:** `Rs − (Rp ‖ C)`

**Parâmetros:** Rs, Rp, C (3 parâmetros — mínimo do sistema)

**Uso principal:** modelo baseline / critério BIC. Se um circuito mais complexo não superar o `Simple-RC` em BIC/AICc, os parâmetros extras não são justificados pelos dados.

**Sistemas típicos:** eletrodos bloqueantes ideais, eletrólitos aquosos simples.

---

### 8. `CPE-Simple`

**Topologia:** `Rs − CPE`

**Parâmetros:** Rs, Q, n (3 parâmetros)

**Significado físico:**
- **Q** — pseudo-capacitância; n → 1 ⇒ capacitor ideal (EDLC perfeito); n = 0,85–0,95 ⇒ EDLC real de carbono ativado.
- **Rs** — ESR (Equivalent Series Resistance).

**Sistemas típicos:** EDLC (supercapacitores), eletrodos bloqueantes de Pt/C vítreo em ácido, filmes de eletrólito polimérico a alta frequência.

**Quando usar:** ausência de qualquer processo faradaico na janela de medição.

---

### 9. `Warburg-Short`

**Topologia:** `Rs − (Rp ‖ CPE) − W_coth`

**Parâmetros:** Rs, Rp, Q, n, Rd, Td (6 parâmetros)

**Diferença de `Warburg-Finite`:** boundary condition refletiva (eletrodo bloqueante no fundo) → coth → a baixas frequências Z tende ao eixo imaginário (comportamento capacitivo), não ao eixo real.

**Sistemas típicos:** células de camada fina com contra-eletrodo bloqueante, eletrodos seletivos de íons, membranas poliméricas, eletrolitos poliméricos de Li.

---

### 10. `Gerischer`

**Topologia:** `Rs − (Rp ‖ CPE) − Z_Ger`

$$Z_{Ger} = \frac{R_g}{\sqrt{1 + j\omega T_g}}$$

**Parâmetros:** Rs, Rp, Q, n, Rg, Tg (6 parâmetros)

**Significado físico:**
- **Rg** — amplitude da impedância da reação distribuída.
- **Tg** = k_f/D — taxa de reação homogênea dividida pela difusividade.

**Diferença do Warburg:** o arco Gerischer começa em ~45° mas curva-se para o eixo real a baixas frequências; o Warburg semi-infinito mantém 45° indefinidamente.

**Sistemas típicos:** catodos SOFC (LSC, LSCF, LSM) — reação de redução de O₂, condutores mistos iônicos-eletrônicos (MIEC), eletrodos de difusão gasosa.

---

### 11. `Three-ZARC`

**Topologia:** `Rs − ZARC₁ − ZARC₂ − ZARC₃`

**Parâmetros:** Rs, R1, Q1, n1, R2, Q2, n2, R3, Q3, n3 (10 parâmetros)

**Física dos arcos:**
- **ZARC₁** (alta frequência) — condução no bulk do grão.
- **ZARC₂** (frequência média) — resistência de contorno de grão.
- **ZARC₃** (baixa frequência) — polarização eletrodo/interface.

**Sistemas típicos:** eletrolitos sólidos garnet (LLZO), NASICON (LAGP, LATP), zircônia estabilizada com ítria (YSZ) em SOFC, vitrocerâmicas ricas em Li.

**Cuidado:** 10 parâmetros → risco elevado de overfitting; exigir boa separação visual dos três arcos no Nyquist antes de usar.

---

## Circuitos especializados (4) — `Porous-Coating-TLM`, `MXene-Intercalation`, `De-Levie-TLM`, `Pseudo-Capacitance-CPE`

> Presentes no registry (fitting) **e** no gerador sintético (treino ML) desde a v0.4.10.
> Anteriormente apenas no registry; dados sintéticos adicionados para completar a cobertura do classificador.

---

### 12. `Porous-Coating-TLM`

**Topologia:** `Rs − (Rcoat ‖ Cpore) − (Rct ‖ CPEdl)`

**Diferença de `Coating-CPE`:** o revestimento usa capacitor ideal (Cpore, n=1), capturando a capacitância geométrica (dielétrica) do filme intacto; apenas a dupla camada usa CPE, refletindo que a rugosidade da interface é muito maior que a do filme.

**Parâmetros:** Rs, Rcoat, Cpore, Rct, Qdl, n_dl (6 parâmetros)

**Sistemas típicos:** revestimentos epóxi/poliuretano em aço após imersão, alumínio anodizado com preenchimento de poros, aço fosfatado, SEI em grafite (modelo 2 camadas simplificado).

---

### 13. `MXene-Intercalation`

**Topologia:** `Rs − (Rsei ‖ CPEsei) − (Rct ‖ CPEdl) − W_fin`

**Parâmetros:** Rs, Rsei, Qsei, n_sei, Rct, Qdl, n_dl, AW, tau_d (9 parâmetros)

**Física das três regiões:**
- Alta ω: camada de terminações superficiais (=O, −OH, −F) → SEI do MXene.
- Média ω: transferência de carga na interface MXene/eletrólito.
- Baixa ω: difusão finita de H⁺/Na⁺ no espaçamento interlamelar 2D.

**Sistemas típicos:** Ti₃C₂Tₓ em H₂SO₄ 1 M (pseudocapacitância de prótons), Nb₂CTₓ em H₂SO₄, Ti₃C₂Tₓ em Na₂SO₄.

---

### 14. `De-Levie-TLM`

**Topologia:** `Rs + √(Ri/Ydl) · coth(L√(Ri·Ydl))`

**Parâmetros:** Rs, Ri, Qdl, n_dl, L (5 parâmetros)

**Física:** modelo de linha de transmissão para eletrodo poroso (poro cilíndrico de comprimento L). Alta ω → linha de 45° no Nyquist; baixa ω → resposta capacitiva vertical.

**Sistemas típicos:** filmes de MXene (canais 2D empilhados), carbono ativado (EDLC), arrays de CNT verticalmente alinhados, aerogel de grafeno.

---

### 15. `Pseudo-Capacitance-CPE`

**Topologia:** `Rs − (Rct ‖ CPEdl) − (Rads ‖ Cads)`

**Parâmetros:** Rs, Rct, Qdl, n_dl, Rads, Cads (6 parâmetros)

**Diferença de `Two-Arc-CPE`:** o segundo arco usa capacitor ideal (Cads, não CPE), pois adsorção por isoterma de Langmuir produz elemento capacitivo ideal no limite linear.

**Sistemas típicos:** RuO₂ em H₂SO₄ (pseudocapacitância de adsorção), MnO₂ em Na₂SO₄, Nb₂O₅ nanocristalino, deposição subpotencial (UPD) de Cu/Pb em Au.

---

## Circuitos estendidos (22) — EXT-01 a EXT-22

> Presentes no `CircuitRegistry` (fitting) **e** no gerador sintético (treino ML) desde a v0.4.10.
> Anteriormente apenas no gerador; templates de fitting adicionados ao registry para que
> o classificador ML possa predizê-los e o pipeline de fitting os ajuste diretamente sem fallback.
> Os 11 circuitos mais complexos (`_COMPLEX_CIRCUITS`) recebem o dobro de amostras sintéticas.

---

### EXT-01 — `Rs-ZARC-TLM`

**Topologia:** `Rs − ZARC − TLM`

**Parâmetros:** Rs, R, Q, n, Ri, Ydl, nt (7 params)

**Física:** arco de transferência de carga em série com linha de transmissão De Levie. Aparece em eletrodos de carbono ativado com reação faradaica superficial superposta à capacitância de dupla camada distribuída.

**Sistemas:** CNTs funcionalizados, carbono ativado com grupos quinona redox, eletrodos compostos de MXene/RGO.

---

### EXT-02 — `Rs-ZARC-ZARC-Wfinite`

**Topologia:** `Rs − ZARC₁ − ZARC₂ − W_fin`

**Parâmetros:** Rs, R1, Q1, n1, R2, Q2, n2, Rd, Td (9 params)

**Física:** dois processos de transferência de carga (e.g. SEI + CT) com difusão finita a baixas frequências. Mais rico que `ZARC-ZARC-W` porque a camada difusiva é limitada.

**Sistemas:** baterias Li-ion com eletrodo de grafite thick (SEI + CT + difusão Li⁺ limitada), óxidos de transição com dois processos de inserção.

---

### EXT-03 — `Rs-ZARC-ZARC-Wshort`

**Topologia:** `Rs − ZARC₁ − ZARC₂ − W_coth`

**Parâmetros:** Rs, R1, Q1, n1, R2, Q2, n2, Rd, Td (9 params)

**Física:** como EXT-02 mas com boundary condition refletiva — a baixas frequências Z sobe verticalmente indicando que o íon não atravessa o eletrodo.

**Sistemas:** membranas de troca iônica com dois processos de interface, eletrodos de inserção com poros fechados.

---

### EXT-04 — `Rs-ZARC-ZARC-Gerischer`

**Topologia:** `Rs − ZARC₁ − ZARC₂ − Ger`

**Parâmetros:** Rs, R1, Q1, n1, R2, Q2, n2, Rg, Tg (9 params)

**Física:** dois arcos de interface seguidos de reação distribuída acoplada a difusão química. Característico de SOFC com catodo de dupla camada (e.g. LSM + YSZ/LSM composto).

**Sistemas:** catodos SOFC com duas microestruturas distintas, catalisadores de ORR multicamadas.

---

### EXT-05 — `Rs-ZARC-ZARC-TLM`

**Topologia:** `Rs − ZARC₁ − ZARC₂ − TLM`

**Parâmetros:** Rs, R1, Q1, n1, R2, Q2, n2, Ri, Ydl, nt (10 params)

**Física:** dois arcos de interface + eletrodo poroso em série. Eletrodos compostos (ativo + condutor poroso) com duas interfaces distintas.

---

### EXT-06 — `Rs-RC-ZARC-W`

**Topologia:** `Rs − (R₁‖C₁) − ZARC₂ − W`

**Parâmetros:** Rs, R1, C1, R2, Q2, n2, sigma (7 params)

**Física:** revestimento com capacitor ideal (filme dielétrico homogêneo) + transferência de carga CPE + Warburg. Útil quando o filme é bem definido e a rugosidade fica concentrada na interface metal/eletrólito.

---

### EXT-07 — `Rs-ZARC-RC-Wfinite`

**Topologia:** `Rs − ZARC₁ − (R₂‖C₂) − W_fin`

**Parâmetros:** Rs, R1, Q1, n1, R2, C2, Rd, Td (8 params)

**Física:** processo CPE a alta frequência + processo ideal (e.g. adsorção com isoterma Langmuir) + difusão finita. Variante de `Pseudo-Capacitance-CPE` com difusão.

---

### EXT-08 — `Rs-L-ZARC-W`

**Topologia:** `Rs − L − ZARC − W`

**Parâmetros:** Rs, L, R, Q, n, sigma (6 params)

**Física:** indutância em série (cabo ou adsorção) + Randles-CPE-W. Frequente em células eletroquímicas com cabos de medição longos combinado com difusão de massa.

---

### EXT-09 — `Rs-L-ZARC-Wfinite`

**Topologia:** `Rs − L − ZARC − W_fin`

**Parâmetros:** Rs, L, R, Q, n, Rd, Td (7 params)

**Física:** variante de EXT-08 com camada difusiva limitada (filme fino ou célula simétrica).

---

### EXT-10 — `Rs-L-ZARC-ZARC`

**Topologia:** `Rs − L − ZARC₁ − ZARC₂`

**Parâmetros:** Rs, L, R1, Q1, n1, R2, Q2, n2 (8 params)

**Física:** loop indutivo + dois processos interfaciais sem difusão. Ocorre em células a combustível PEM com dois arcos de catodo/anodo e artefato indutivo a alta frequência.

---

### EXT-11 — `Rs-ZARC-ZARC-ZARC-W`

**Topologia:** `Rs − ZARC₁ − ZARC₂ − ZARC₃ − W`

**Parâmetros:** Rs, R1, Q1, n1, R2, Q2, n2, R3, Q3, n3, sigma (11 params)

**Física:** extensão de `Three-ZARC` com cauda de difusão. Eletrolitos sólidos com eletrodo poroso em contato.

**Cuidado:** 11 parâmetros — alta probabilidade de overfitting. Usar apenas com dados de alta qualidade e ampla faixa de frequência.

---

### EXT-12 — `Rs-ZARC-ZARC-ZARC-Wfinite`

**Topologia:** `Rs − ZARC₁ − ZARC₂ − ZARC₃ − W_fin`

**Parâmetros:** 12 params

**Física:** como EXT-11 mas com difusão finita. Modelo mais completo para células sólidas full-cell com eletrodo poroso e camada difusiva limitada.

---

### EXT-13 — `Rs-ZARC-CPE`

**Topologia:** `Rs − ZARC − CPE`

**Parâmetros:** Rs, R, Q1, n1, Q2, n2 (6 params)

**Física:** processo de CT + camada de bloqueio capacitivo a baixas frequências (e.g. filmes passivantes parcialmente bloqueantes). O CPE final não tem resistência paralela → impede fluxo DC.

---

### EXT-14 — `Rs-RC-W`

**Topologia:** `Rs − (R‖C) − W`

**Parâmetros:** Rs, R, C, sigma (4 params)

**Física:** versão ideal de `Randles-CPE-W` com n=1. Superfícies altamente lisas ou quando a qualidade dos dados não justifica um parâmetro n adicional.

---

### EXT-15 — `Rs-RC-Wfinite`

**Topologia:** `Rs − (R‖C) − W_fin`

**Parâmetros:** Rs, R, C, Rd, Td (5 params)

**Física:** versão ideal de `Warburg-Finite`. Modelo parcimonioso para filmes finos com superfície homogênea.

---

### EXT-16 — `Rs-ZARC-ZARC-CPE`

**Topologia:** `Rs − ZARC₁ − ZARC₂ − CPE`

**Parâmetros:** Rs, R1, Q1, n1, R2, Q2, n2, Q3, n3 (9 params)

**Física:** dois arcos de transferência de carga + camada capacitiva bloqueante a baixas frequências. Sistemas onde a difusão é suprimida por uma camada passivante espessa (e.g. SEI muito resistente).

---

### EXT-17 — `Rs-RC-ZARC-Wfinite`

**Topologia:** `Rs − (R₁‖C₁) − ZARC₂ − W_fin`

**Parâmetros:** Rs, R1, C1, R2, Q2, n2, Rd, Td (8 params)

**Física:** revestimento dielétrico ideal + CT rugoso + difusão finita. Variante de `Coating-CPE` com difusão explícita — útil para revestimentos protetores sobre eletrodos de intercalação.

---

### EXT-18 — `Rs-ZARC-RC-Wshort`

**Topologia:** `Rs − ZARC₁ − (R₂‖C₂) − W_coth`

**Parâmetros:** Rs, R1, Q1, n1, R2, C2, Rd, Td (8 params)

**Física:** processo CPE a alta frequência + adsorção ideal + difusão refletiva. Sistemas com adsorção de intermediários + camada difusiva com boundary bloqueante.

---

### EXT-19 — `Rs-L-ZARC-ZARC-W`

**Topologia:** `Rs − L − ZARC₁ − ZARC₂ − W`

**Parâmetros:** Rs, L, R1, Q1, n1, R2, Q2, n2, sigma (9 params)

**Física:** loop indutivo + dois arcos + Warburg. PEM fuel cells com dois processos de eletrodo e cabos de medição.

---

### EXT-20 — `Rs-TLM`

**Topologia:** `Rs − TLM`

**Parâmetros:** Rs, Ri, Ydl, nt (4 params)

**Física:** linha de transmissão De Levie pura sem arco de transferência de carga — eletrodo puramente capacitivo com estrutura porosa (EDLC ideal poroso).

---

### EXT-21 — `Rs-ZARC-ZARC-ZARC-Gerischer`

**Topologia:** `Rs − ZARC₁ − ZARC₂ − ZARC₃ − Ger`

**Parâmetros:** Rs, R1, Q1, n1, R2, Q2, n2, R3, Q3, n3, Rg, Tg (12 params)

**Física:** três processos de interface + reação distribuída acoplada a difusão química. Modelo mais completo para SOFC cerâmicas policristalinas com catodo ativo.

**Cuidado:** 12 parâmetros — uso restrito a espectros com pelo menos 4 arcos claramente distinguíveis.

---

### EXT-22 — `Rs-ZARC-TLM-W`

**Topologia:** `Rs − ZARC − TLM − W`

**Parâmetros:** Rs, R, Q, n, Ri, Ydl, nt, sigma (8 params)

**Física:** arco de CT + eletrodo poroso + cauda difusiva semi-infinita. Supercapacitores assimétricos com componente faradaica + eletrodo de alta área superficial com difusão de íons além dos poros.

---

## Recomendações de melhoria

### ✅ 1. Adicionar os 4 circuitos do registry ao gerador sintético *(concluído — v0.4.10)*

`Porous-Coating-TLM`, `MXene-Intercalation`, `De-Levie-TLM` e `Pseudo-Capacitance-CPE`
adicionados a `gen_synthetic_eis.py` com ranges físicos alinhados ao `circuit_registry.py`.
O classificador ML agora reconhece todos os 37 circuitos.

---

### ✅ 2. Adicionar templates de fitting para os 22 circuitos EXT no registry *(concluído — v0.4.10)*

Todos os 22 circuitos EXT registados em `src/circuit_registry.py` com modelos matemáticos
consistentes com `gen_synthetic_eis.py`. O pipeline de fitting já não cai em fallback
quando o classificador ML prediz um circuito EXT.

---

### 3. Separar famílias de circuitos para classificação hierárquica

**Problema atual:** o classificador RandomForest trata todos os 33 circuitos como classes independentes. Circuitos muito similares (e.g. `Warburg-Finite` vs. `Warburg-Short`, ou `EXT-02` vs. `EXT-03`) têm EIS quase idêntico para muitas combinações de parâmetros, causando confusão no classificador.

**Melhoria proposta:** implementar classificação em dois estágios:
1. **Família** (6–8 grupos): Randles simples / ZARC múltiplo / com difusão / indutivo / poroso / Gerischer.
2. **Circuito específico** dentro da família.

Isso reduz a confusão inter-classe e aumenta a precisão geral sem aumentar o dataset.

---

### 4. Aumentar diversidade do dataset sintético com ruído heteroscedástico

**Problema atual:** `gen_synthetic_eis.py` adiciona ruído gaussiano homoscedástico (`noise_level`). Dados reais apresentam ruído proporcional ao módulo de Z (heteroscedástico) + outliers ocasionais.

**Melhoria proposta:**
```python
# Ruído multiplicativo + outliers (5% dos pontos)
noise_real = z_real * rng.normal(0, noise_level) + rng.normal(0, noise_level * 0.1)
noise_imag = z_imag * rng.normal(0, noise_level) + rng.normal(0, noise_level * 0.1)
```

---

### ✅ 5. Incluir validação de Kramers-Kronig como feature de entrada do classificador *(concluído — v0.4.10)*

Adicionadas 3 features KK ao vetor de entrada do classificador (12 features no total):
`kk_residual_real`, `kk_residual_imag`, `kk_valid`. Calculadas por `extract_eis_features_for_ml()`
em `src/circuit_fitting.py` e usadas por `CircuitMLSelector` e `HierarchicalCircuitSelector`.

---

### 6. Balancear o dataset para circuitos raros

**Problema:** com 20 arquivos por circuito, classes com espectro muito típico (e.g. `Simple-RC`) dominam a fronteira de decisão. Circuitos raros com espectros atípicos (e.g. `Three-ZARC`, `EXT-21`) são subestimados.

**Melhoria:** usar `class_weight='balanced'` no RandomForest ou SMOTE para oversample circuitos com baixa frequência de ocorrência no dataset real. Para o dataset sintético, aumentar para 40–50 arquivos por circuito complexo (>8 parâmetros).

---

### ✅ 7. Adicionar intervalo de confiança ao output do classificador *(concluído — v0.4.10)*

`CalibratedClassifierCV(method='sigmoid')` aplicado em torno do `RandomForestClassifier`
em ambos `CircuitMLSelector` e `HierarchicalCircuitSelector`. As probabilidades retornadas
pelo método `predict()` somam 1.0 e são calibradas (Platt scaling).

---

### ✅ 8. Exportar os metadados de circuito no relatório PDF *(concluído — v0.4.10)*

Secção "Equivalent Circuit" adicionada ao `ReportGenerator` (PDF e Markdown):
- Diagrama ASCII do circuito (`template.diagram`).
- Tabela de significado físico dos parâmetros (`template.physical_meaning`).
- Sistemas típicos (`template.typical_systems`).
Activada automaticamente em `gui_app.py` via `_circuit_info` passado a `gen.generate()`.

---

*Gerado em: 2026-05-22 — IonFlow Pipeline v0.4.10*
