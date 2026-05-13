# IonFlow Pipeline — Commercial Roadmap
**Objetivo: tornar o IonFlow a referência de mercado em análise de EIS, com fabricantes de potenciostatos querendo acoplar ou licenciar o software.**

> Versão atual: **v0.4.0** — Market Edition lançada.

---

## Situação competitiva

| Produto | Preço | DRT | Batch | Formatos | Status |
|---------|-------|-----|-------|----------|--------|
| **IonFlow** | OSS (licença em desenvolvimento) | ✅ nativo | ✅ | BioLogic, Gamry, Zahner, Autolab | ativo |
| RelaxIS | ~€890 one-time | ❌ | ❌ | proprietário | ativo |
| EIS Spectrum Analyser | grátis | ❌ | ❌ | limitado | abandonado |
| ZView | ~$2 500 | ❌ | ❌ | proprietário | stagnado |
| EC-Lab / NOVA | bundle com hardware | parcial | ❌ | apenas próprio | ativo (locked-in) |

**Vantagem principal do IonFlow:** único produto a combinar DRT nativo + batch processing + importação multi-marca + análise IA + relatório PDF + open-source core.

---

## v0.4.0 — "Market Edition" _(próximos 3–4 meses)_

**Foco:** remover todas as barreiras de entrada para laboratórios acadêmicos e industriais.

### 0.4.x — Estabilidade e Polimento

| ID | Feature | Impacto | Esforço |
|----|---------|---------|---------|
| MKT-01 | Adicionar `galvani` ao `pip install ionflow-pipeline[biologic]` | ★★★ | XS | ✅ |
| MKT-02 | **Branding PDF**: logo + instituição configuráveis na aba Configurações | ★★★ | S | ✅ |
| MKT-03 | Preview do logo no relatório antes de exportar | ★★ | M | ✅ |
| MKT-04 | Suporte a `.idf` / `.dfr` (Solartron) | ★★ | M | ✅ |
| MKT-05 | Exportação para `.xlsx` com múltiplas abas (EIS + DRT + ciclagem) | ★★★ | S | ✅ |
| MKT-06 | Validação KK com score visual (semáforo) na aba principal | ★★ | S | ✅ |
| MKT-07 | Assinatura digital do relatório PDF (hash SHA-256 no rodapé) | ★ | XS | ✅ |
| MKT-08 | Internacionalização completa: todas as strings pendentes em `i18n_strings/` | ★★ | M | ✅ |

> **MKT-01 e MKT-02 já implementados em v0.3.1+.**

### 0.4.x — Sistema de Licença (Freemium → Pro)

| ID | Feature | Descrição |
|----|---------|-----------|
| LIC-01 | **Modo Free**: 5 arquivos/sessão, sem PDF branding personalizado | Padrão sem ativação | ✅ |
| LIC-02 | **Modo Pro** (licença por e-mail): ilimitado + branding + API | €149/ano | ✅ |
| LIC-03 | **Modo Lab** (5 usuários): dashboard compartilhado + cloud sync | €499/ano | ✅ |
| LIC-04 | **Modo OEM** (embed): SDK Python + suporte dedicado | contrato | ✅ |
| LIC-05 | Servidor de validação leve (Flask, auto-hospedado ou SaaS) | infra mínima | ✅ |

```
Fluxo de ativação:
1. Usuário compra em ionflow.io (Stripe/Paddle)
2. Recebe license_key por e-mail
3. Cola na aba Configurações > Licença
4. Cliente valida contra servidor HTTPS (hash HMAC + UUID hardware)
5. Offline grace period: 30 dias
```

---

## v0.5.0 — "Lab Edition" _(6–9 meses)_

**Foco:** multi-usuário, colaboração, integração com sistemas de gestão de laboratório (LIMS).

| ID | Feature | Impacto |
|----|---------|---------|
| LAB-01 | Dashboard Streamlit para acesso web (`pip install ionflow[dashboard]`) | ★★★ |
| LAB-02 | Banco de dados SQLite/PostgreSQL de amostras e resultados | ★★★ |
| LAB-03 | REST API (`/api/v1/analyze`, `/api/v1/report`) para integração de instrumento | ★★★ |
| LAB-04 | Exportação para ELN (Electronic Lab Notebook): RSpace, LabArchives, Benchling | ★★ |
| LAB-05 | Sincronização de resultados entre instâncias (rsync / S3) | ★★ |
| LAB-06 | Plugin de importação automática: monitorar pasta e processar ao detectar arquivo novo | ★★★ |
| LAB-07 | Suporte a JSON-LD para metadados FAIR (findable, accessible, interoperable, reusable) | ★ |

---

## v0.6.0 — "OEM / Manufacturer SDK" _(12–18 meses)_

**Foco:** fabricantes de potenciostatos querem acoplar o IonFlow ao seu software proprietário.

### SDK Python

```python
# Exemplo de uso pelo fabricante
from ionflow_sdk import EISAnalyzer, ReportBuilder

analyzer = EISAnalyzer(license_key="OEM-GAMRY-XXXX")
result = analyzer.analyze(freq=freq, zreal=zreal, zimag=zimag)

report = ReportBuilder(logo="gamry_logo.png", institution="Gamry Instruments")
report.from_result(result)
report.save("analysis.pdf")
```

| ID | Feature | Descrição |
|----|---------|-----------|
| OEM-01 | `ionflow-sdk` PyPI package (subset sem GUI) | core analítico puro |
| OEM-02 | White-label: remover branding IonFlow, substituir por marca do parceiro | necessário para acordos |
| OEM-03 | C extension / DLL wrapper para integração com software em C++/C# | Gamry, Metrohm, Solartron |
| OEM-04 | Documentação de API REST OpenAPI 3.0 | parceiros enterprise |
| OEM-05 | Acordo de redistribuição (royalty por instrumento vendido ou annual flat fee) | €5–15 por unidade ou €5 000+/ano |

### Parceiros alvo (por prioridade)

1. **PalmSens** (Países Baixos) — software EmStat tem análise fraca; maior abertura para OSS
2. **Metrohm Autolab** (Suíça) — NOVA software tem DRT rudimentar; nosso DRT é superior
3. **BioLogic** (França) — EC-Lab não tem batch; porém são conservadores
4. **Gamry** (EUA) — Echem Analyst é datado; abertos a parcerias acadêmicas
5. **WonATech** (Coreia) — Crescendo em ESS/bateria; pouco software de análise
6. **Zahner** (Alemanha) — Thales software; clientes acadêmicos são nosso público

---

## v1.0.0 — "Enterprise" _(18–24 meses)_

**Marco:** produto maduro, auditável, certificável para uso em QC industrial.

| ID | Feature |
|----|---------|
| ENT-01 | Certificação ISO/IEC 17025 (traçabilidade de resultados, audit trail) |
| ENT-02 | Relatórios com assinatura digital qualificada (eIDAS compliant) |
| ENT-03 | Modo offline air-gapped (fábricas com restrição de rede) |
| ENT-04 | Suporte a 21 CFR Part 11 (farmacêutico, FDA) |
| ENT-05 | Integração SCADA/MES via OPC-UA |
| ENT-06 | Multi-tenancy cloud (SaaS) com isolamento de dados por cliente |

---

## Modelo de negócio

### Precificação sugerida

| Plano | Preço | Usuários | Features |
|-------|-------|----------|----------|
| **Community** | Grátis | 1 | Sem limite arquivos, sem branding, sem suporte |
| **Pro** | €149/ano | 1 | Branding PDF, API local, suporte por e-mail 5 dias úteis |
| **Lab** | €499/ano | 5 | Pro + dashboard web, banco de dados, sincronização |
| **Institution** | €1 200/ano | ilimitado | Lab + SLA 24h, onboarding, treinamento 2h |
| **OEM** | €5 000/ano base | embed | SDK, white-label, suporte prioritário, NDA |

### Canais de venda

1. **ionflow.io** — site direto com Paddle/Stripe (mais simples que Stripe para PT/EU)
2. **GitHub Sponsors** — tier Pro/Lab via GitHub
3. **Zenodo DOI** — citações científicas aumentam visibilidade orgânica
4. **ResearchGate / Academia.edu** — demo paper + link de download
5. **Distribuição via fabricantes** (OEM) — royalty passivo

### CAC estimado (Customer Acquisition Cost)

- Acadêmico: €0 (descoberta orgânica via GitHub/paper)
- Industrial Lab: €50–150 (LinkedIn Ads, conference booths)
- OEM: €2 000–5 000 (vendas enterprise, demos presenciais)

---

## Marketing e visibilidade

### Curto prazo (1–2 meses)

- [ ] **Landing page** em ionflow.io (GitHub Pages ou Vercel, custo €0)
  - Demo GIF de 30 seg mostrando Nyquist → DRT → PDF
  - Comparação com RelaxIS/ZView em 3 bullets
  - CTA: "Download Free" + "Buy Pro"
- [ ] **Paper no JOSS** (Journal of Open Source Software) — revisão em 2–4 semanas, DOI garante citações
- [ ] **Post no r/electrochemistry** + ResearchGate
- [ ] **Demo video** 2 min no YouTube (screen recording + narração PT e EN)

### Médio prazo (3–6 meses)

- [ ] Apresentar em **ECS (Electrochemical Society)** ou **ISE (International Society of Electrochemistry)**
- [ ] Contato direto com PalmSens e Metrohm para demo técnica
- [ ] Parceria com grupos de pesquisa em baterias de sódio / supercapacitores (co-autoria em paper)

### Longo prazo

- [ ] Certificação como ferramenta recomendada pela IEC TC 69 (baterias veiculares)
- [ ] Integração com plataformas de Materials Informatics (Citrine, Materials Project)

---

## Diferenciadores defensáveis

1. **DRT nativo** (único produto open-source com DRT integrado à GUI)
2. **Parser multi-potenciostato** sem conversão manual (BioLogic, Gamry, Zahner, Autolab)
3. **IA interpretativa** com Ollama local (sem dados saindo para cloud)
4. **Batch de centenas de arquivos** com relatório consolidado
5. **Open-source core** (MIT) + camada proprietária (licença Pro/OEM)

---

## Riscos e mitigações

| Risco | Probabilidade | Mitigação |
|-------|--------------|-----------|
| BioLogic/Gamry lança feature equivalente | Média | Focar no que eles nunca farão: OSS, multi-marca, IA local |
| Concorrente copia o código (MIT) | Alta | O valor está no suporte, integração e roadmap — não no código |
| Adoção lenta em mercado conservador | Alta | Paper JOSS + 3 grupos de pesquisa como early adopters/referências |
| Dependência de galvani (sem manutenção ativa) | Baixa | Manter fork interno + parser próprio como backup |

---

## Próximos 30 dias — Ações imediatas

- [x] v0.3.1 lançado com GitHub Release e auto-updater
- [x] Branding PDF (logo + instituição) implementado na Settings tab
- [x] `galvani` adicionado como dependência opcional `[biologic]`
- [ ] Criar landing page no GitHub Pages (1 dia de trabalho)
- [ ] Submeter ao JOSS (preparar `paper.md` de ~500 palavras + `paper.bib`)
- [ ] Implementar sistema de licença básico (LIC-01 a LIC-03)
- [ ] Gravar demo video de 2 min
- [ ] Contato inicial com PalmSens (e-mail direto ao CTO)

---

*Documento criado em: v0.3.1+ | Revisar a cada release minor*
