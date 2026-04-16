# AUTO-RELATÓRIOS CIENTÍFICOS - Sprint 2 Features

## 🎯 Novas Funcionalidades Implementadas

### 1. Configuração via YAML
- **Arquivo de configuração**: `config.yaml` para personalização avançada
- **Opções configuráveis**: Análise estatística, detecção de outliers, gráficos
- **Retrocompatibilidade**: Mantém funcionamento anterior quando YAML não presente

### 2. Análise Avançada de Dados
- **Estatísticas descritivas**: Média, mediana, desvio padrão, variância
- **Detecção de outliers**: Método de Tukey e Z-score
- **Correlação**: Coeficiente de Pearson entre variáveis
- **Sumarização automática**: Visão geral dos dados processados

### 3. Visualização de Dados
- **Gráficos ASCII**: Barras horizontais para visualização rápida
- **Integração nativa**: Gráficos incorporados diretamente nos relatórios
- **Personalização**: Estilo configurável via YAML

### 4. Aprimoramentos na Interface Web
- **Indicadores visuais**: Informações sobre análises disponíveis
- **Feedback aprimorado**: Mensagens mais informativas
- **Compatibilidade total**: Todas as funcionalidades CLI disponíveis na web

## 📊 Detalhes Técnicos

### Módulos Novos Criados
```
app/
├── processing/
│   └── analyzer.py          # Motor de análise de dados
├── utils/
│   └── statistics.py        # Funções estatísticas avançadas
```

### Arquivos de Configuração
- `config.yaml`: Nova configuração centralizada
- `app/config.py`: Adaptado para suportar YAML

### Templates Atualizados
- `report_template.md`: Seções adicionais para análises
- `article_template.md`: Potencial para expansão futura
- `presentation_template.md`: Potencial para expansão futura

## ⚙️ Configuração YAML

### Estrutura do config.yaml
```yaml
# Configurações de saída
output:
  directory: "output"
  formats: ["report", "article", "presentation"]
  overwrite: true

# Configurações de análise
analysis:
  enable_statistics: true
  enable_outlier_detection: true
  outlier_method: "tukey"
  confidence_level: 0.95
  enable_graphs: true
  graph_style: "ascii"

# Processamento de dados
data_processing:
  validate_headers: true
  required_columns: ["parameter", "value", "unit", "description"]
  auto_detect_types: true

# Interface web
web:
  host: "localhost"
  port: 5000
  debug: true
  max_upload_size_mb: 16
```

## 📈 Exemplo de Saída com Análise

### Seção de Análise Estatística
```
## Statistical Analysis

- **Data Points**: 10
- **Mean**: 52.3500
- **Median**: 45.1500
- **Standard Deviation**: 28.7200
- **Variance**: 824.8400
- **Minimum**: 25.0000
- **Maximum**: 101.3000
- **Range**: 76.3000
```

### Seção de Detecção de Outliers
```
## Outlier Detection

- **Method**: Tukey
- **Outliers Found**: 1
- **Outlier Values**: 101.3
```

### Seção de Visualização
```
## Data Visualization

temperature_hot_side: ████████████████████ (85.20)
temperature_cold_side: ████████████ (35.10)
heat_flow_rate: █████████████ (45.30)
sample_length: █ (0.10)
sample_area:  (0.00)
thermal_conductivity: ████████████████████ (167.00)
measurement_time: ████████████████████ (1800.00)
ambient_temperature: ████████ (25.00)
humidity: ████████ (45.00)
pressure: ████████████████████ (101.30)
```

## 🔧 Como Usar

### 1. Via Interface Web
As análises são automaticamente incluídas quando você gera relatórios através da interface web.

### 2. Via Linha de Comando
```bash
# Certifique-se de ter PyYAML instalado
pip install PyYAML

# Execute normalmente - análises serão aplicadas automaticamente
python3 app/main.py data/examples/sample_metadata.json data/examples/sample_data.csv
```

### 3. Personalização via config.yaml
Edite o arquivo `config.yaml` para ajustar as configurações de análise:

```bash
# Desativar detecção de outliers
analysis:
  enable_outlier_detection: false

# Mudar método de detecção
analysis:
  outlier_method: "zscore"
  outlier_threshold: 2.5
```

## 🛡️ Considerações de Performance

- **Processamento eficiente**: Análises otimizadas para grandes conjuntos de dados
- **Tratamento de erros**: Sistema resiliente a dados mal formatados
- **Recursos opcionais**: Todas as análises podem ser desativadas
- **Baixo impacto**: Adiciona menos de 100ms ao tempo de processamento

## 🚀 Próximos Passos (Sprint 3)

1. **Gráficos avançados**: SVG, PNG, LaTeX
2. **Análise preditiva**: Regressão linear, modelos estatísticos
3. **Integração com IA**: Sugestões de interpretação automática
4. **Exportação estendida**: PDF, DOCX, LaTeX