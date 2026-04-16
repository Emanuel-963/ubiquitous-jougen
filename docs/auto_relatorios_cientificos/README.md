# AUTO-RELATÓRIOS CIENTÍFICOS

Sistema automático para geração de relatórios científicos, artigos e apresentações a partir de dados experimentais.

## 🎯 Objetivo

Automatizar a criação de documentos científicos com qualidade profissional, permitindo que estudantes e pesquisadores foquem na ciência em vez de formatação.

## 🚀 Novidade Sprint 1 - Interface Web MVP

Implementamos uma interface web intuitiva que permite usar o sistema sem conhecimento técnico em programação! Agora você pode:

- Upload de arquivos via interface gráfica
- Pré-visualização dos dados antes da geração
- Seleção de templates com clique
- Download automático dos documentos gerados

## 🚀 Funcionalidades

- Geração automática de relatórios científicos completos
- Criação de artigos resumidos
- Preparação de roteiros de apresentação
- Processamento de dados experimentais (CSV)
- Metadados estruturados (JSON)
- Templates personalizáveis
- Sistema modular e extensível
- Sem dependências externas para o MVP

## 📁 Estrutura do Projeto

```
auto_relatorios_cientificos/
├── app/                    # Código-fonte principal
│   ├── main.py             # Ponto de entrada CLI
│   ├── web.py              # Ponto de entrada Web (Sprint 1)
│   ├── config.py           # Configurações
│   ├── io/                 # Leitura/escrita de arquivos
│   ├── processing/         # Processamento de dados
│   ├── generators/         # Geradores de conteúdo
│   ├── templates/          # Templates de documentos
│   ├── web_templates/      # Templates HTML (Sprint 1)
│   ├── static/             # Arquivos estáticos (Sprint 1)
│   └── utils/              # Funções utilitárias
├── data/
│   └── examples/           # Exemplos de dados
├── output/                 # Diretório de saída (gerado)
├── README.md               # Este arquivo
├── requirements.txt        # Dependências
├── MVP_SPRINT1.md          # Documentação Sprint 1
└── .gitignore             # Arquivos ignorados pelo Git
```

## 🛠️ Instalação

1. Clone o repositório:
```bash
git clone <url-do-repositorio>
cd auto_relatorios_cientificos
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

> **Nota**: Para a interface web (Sprint 1 MVP), é necessário o Flask. A interface de linha de comando funciona apenas com a biblioteca padrão.

## ▶️ Uso

### Interface de Linha de Comando (Método Original)

Execute o programa com os arquivos de metadados e dados:

```bash
python3 app/main.py data/examples/sample_metadata.json data/examples/sample_data.csv
```

Opcionalmente, especifique um diretório de saída diferente:

```bash
python3 app/main.py data/examples/sample_metadata.json data/examples/sample_data.csv -o meu_diretorio_saida
```

### Interface Web (Sprint 1 MVP) - Recomendado para Iniciantes

Inicie a interface web:

```bash
# Primeiro ative o ambiente virtual (se necessário)
source venv/bin/activate

# Inicie o servidor web
python3 app/web.py
```

Em seguida, acesse `http://localhost:5000` no seu navegador para usar a interface gráfica com as seguintes funcionalidades:

- **Upload Intuitivo**: Arraste e solte seus arquivos de metadados (JSON) e dados (CSV)
- **Pré-visualização**: Veja seus dados antes de gerar os documentos
- **Seleção de Templates**: Escolha entre diferentes formatos de saída
- **Download Automático**: Receba seus documentos prontos para uso

#### Recursos da Interface Web:
- ✅ Totalmente responsiva (funciona em desktop e mobile)
- ✅ Validação automática de arquivos
- ✅ Feedback visual em tempo real
- ✅ Zero conhecimento técnico necessário
- ✅ Mesma qualidade de saída do método CLI

## 📄 Formato dos Arquivos de Entrada

### Metadados (JSON)

O arquivo de metadados deve conter informações sobre o experimento:

```json
{
  "title": "Título do Experimento",
  "authors": ["Autor 1", "Autor 2"],
  "date": "2026-04-11",
  "objective": "Objetivo do experimento",
  "methodology": "Descrição da metodologia utilizada",
  "parameters": {
    "parametro1": "valor1",
    "parametro2": "valor2"
  }
}
```

### Dados (CSV)

O arquivo de dados deve conter medições experimentais:

```csv
parameter,value,unit,description
temperatura,25.0,°C,Temperatura ambiente
pressão,101.3,kPa,Pressão atmosférica
```

## 📤 Saída Gerada

O sistema gera três tipos de documentos:

1. **relatorio.md** - Relatório científico completo com todas as seções
2. **artigo.md** - Artigo resumido para publicações rápidas
3. **apresentacao.md** - Roteiro de apresentação estruturado

## 🔧 Personalização

### Templates

Os templates podem ser modificados em `app/templates/`:
- `report_template.md` - Template para relatórios completos
- `article_template.md` - Template para artigos resumidos
- `presentation_template.md` - Template para apresentações

### Parâmetros

Modifique `app/config.py` para alterar:
- Diretórios padrão
- Formatos suportados
- Campos obrigatórios nos metadados

## 🔄 Extensibilidade Futura

O sistema foi projetado para expansões futuras:

- **Formatos de saída**: LaTeX, PDF, DOCX
- **Análise avançada**: Gráficos, estatísticas
- **Integração com IA**: Melhorias no texto gerado
- **Múltiplos domínios**: Física, química, biologia
- **Interface web**: Frontend para fácil uso

## 📋 Requisitos

- Python 3.6+
- Nenhuma dependência externa para o MVP

## 📝 Licença

Este projeto é disponibilizado como software livre.

## 👥 Contribuições

Contribuições são bem-vindas! Siga estas etapas:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 🆘 Suporte

Para relatar problemas ou pedir ajuda, abra uma issue no repositório.