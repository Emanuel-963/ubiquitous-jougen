# Wizard do Scientific Protocol — IonFlow v0.5.1

## Acesso

No GUI principal, abra **Ferramentas > Protocolo Científico Multicritério**.
Também é possível usar `Ctrl+K` e pesquisar pelo mesmo nome.

## Fluxo

1. Selecione **Nova análise** ou **Carregar análise existente**.
2. Informe nome do projeto, material, eletrólito e célula.
3. Adicione os CVs informando a velocidade em `mV/s`; o wizard converte para `V/s`.
4. Selecione explicitamente o GCD e os estados EIS inicial, pós-CV e final.
5. Informe massa total em `mg`, área em `cm²`, massas dos eletrodos e frações de potencial.
6. Monte visualmente a sequência GCD com as ações de adicionar, remover e mover etapas.
7. Revise os dados e escolha **Salvar sem executar** ou **Salvar e executar**.

Na revisão final, escolha a resolução das figuras: `150`, `200`, `300` ou
`600 DPI`. O padrão é `300 DPI`, recomendado para relatórios e artigos.

## Projetos e replicatas

Projetos são salvos em:

```text
scientific_protocol_projects/<nome_do_projeto>/project.json
```

O arquivo usa `schema_version: "2.0"` e agrupa células por eletrólito:

```json
{
  "schema_version": "2.0",
  "project": {"name": "Estudo Nb2L", "material": "Nb2L"},
  "electrolytes": {
    "NaCl 1M": {
      "cells": {
        "cell_001": {
          "replicate": 1,
          "cv": {},
          "gcd": "",
          "mass_g": 0.0056,
          "cell_area_cm2": 1.0,
          "current_sequence_a_g": [],
          "electrodes": {},
          "eis": {}
        }
      }
    }
  }
}
```

Ao duplicar uma célula, parâmetros físicos, área e sequência são copiados, mas
os arquivos experimentais são removidos para que a nova replicata seja
associada manualmente aos seus próprios dados. A exclusão de uma célula sempre
exige confirmação e nunca deixa um eletrólito sem célula.

## Compatibilidade

O formato antigo com uma entrada direta em `electrolytes` continua aceito pelo
executor. Ao carregar esse formato no wizard, ele é migrado em memória para
`cell_001`; o arquivo original não é sobrescrito até o usuário salvar uma nova
versão.

Os caminhos experimentais são gravados relativos ao diretório do projeto,
quando possível, e resolvidos novamente ao abrir. Isso permite mover o projeto
com seus dados para outro computador.