#!/usr/bin/env python3
"""
Interface web para AUTO-RELATÓRIOS CIENTÍFICOS
Sprint 1 MVP - Interface web simples com pré-visualização
Sprint 2 - Advanced analysis and configuration
"""

import sys
import os
import json
import tempfile
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename

# Adicionar o diretório raiz ao path para importar módulos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.io.reader import read_metadata, read_data
from app.io.writer import write_multiple_outputs
from app.processing.data_processor import structure_experiment_data
from app.processing.analyzer import analyze_data, create_ascii_bar_chart
from app.generators.report_generator import fill_report_template
from app.generators.article_generator import fill_article_template
from app.generators.presentation_generator import fill_presentation_template
from app.utils.validator import validate_metadata_fields
from app.config import OUTPUT_DIR, TEMPLATES_DIR, TEMPLATE_FILES, CONFIG

# Configuração do Flask
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web_templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = 'auto-relatorios-cientificos-sprint1-mvp'  # Chave secreta para sessões
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite de 16MB para uploads

# Diretório temporário para armazenar arquivos durante a sessão
TEMP_DIR = tempfile.mkdtemp()

def load_template(template_name: str) -> str:
    """Carrega template do conteúdo do arquivo"""
    template_file = TEMPLATE_FILES.get(template_name)
    if not template_file:
        raise ValueError(f"Template desconhecido: {template_name}")

    template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", template_file)

    with open(template_path, 'r', encoding='utf-8') as file:
        return file.read()

def generate_reports_from_data(metadata, data, template_choice="all", output_dir=OUTPUT_DIR):
    """Gera relatórios a partir de dados já carregados"""
    # Validar metadados
    is_valid, errors = validate_metadata_fields(metadata)
    if not is_valid:
        raise ValueError(f"Metadados inválidos: {'; '.join(errors)}")

    # Estruturar dados
    structured_data = structure_experiment_data(metadata, data)

    # Adicionar visualizações se habilitadas
    if CONFIG.get("analysis", {}).get("enable_graphs", True):
        try:
            # Extrair valores numéricos para gráfico
            numeric_values = []
            labels = []
            for row in data:
                if "value" in row and "parameter" in row:
                    try:
                        numeric_values.append(float(row["value"]))
                        labels.append(row["parameter"][:10])  # Limitar tamanho do label
                    except (ValueError, TypeError):
                        continue

            # Criar gráfico ASCII
            if numeric_values:
                chart = create_ascii_bar_chart(numeric_values[:10], labels[:10])  # Limitar a 10 itens
                structured_data["metadata"]["data_visualization"] = chart
        except Exception as e:
            structured_data["metadata"]["data_visualization"] = f"Error creating visualization: {str(e)}"

    # Carregar templates
    report_template = load_template("report")
    article_template = load_template("article")
    presentation_template = load_template("presentation")

    # Gerar conteúdo
    outputs = {}

    if template_choice in ["all", "report"]:
        report_content = fill_report_template(structured_data, report_template)
        outputs["relatorio.md"] = report_content

    if template_choice in ["all", "article"]:
        article_content = fill_article_template(structured_data, article_template)
        outputs["artigo.md"] = article_content

    if template_choice in ["all", "presentation"]:
        presentation_content = fill_presentation_template(structured_data, presentation_template)
        outputs["apresentacao.md"] = presentation_content

    # Escrever saídas
    filepaths = write_multiple_outputs(outputs, output_dir)
    return filepaths

@app.route('/')
def index():
    """Página inicial com formulário de upload"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Processa o upload de arquivos e mostra pré-visualização"""
    try:
        # Verificar se os arquivos foram enviados
        if 'metadata' not in request.files or 'data' not in request.files:
            return render_template('index.html', error="Por favor, selecione ambos os arquivos (metadados e dados).")

        metadata_file = request.files['metadata']
        data_file = request.files['data']
        template_choice = request.form.get('template', 'all')
        ai_enhancement = request.form.get('ai_enhancement', 'disabled')

        # Verificar se os arquivos têm nomes
        if metadata_file.filename == '' or data_file.filename == '':
            return render_template('index.html', error="Por favor, selecione ambos os arquivos.")

        # Salvar arquivos temporariamente
        metadata_filename = secure_filename(metadata_file.filename)
        data_filename = secure_filename(data_file.filename)

        metadata_path = os.path.join(TEMP_DIR, metadata_filename)
        data_path = os.path.join(TEMP_DIR, data_filename)

        metadata_file.save(metadata_path)
        data_file.save(data_path)

        # Ler e validar dados
        metadata = read_metadata(metadata_path)
        data = read_data(data_path)

        # Converter para JSON para passar para o template
        metadata_json = json.dumps(metadata)
        data_json = json.dumps(data)

        # Renderizar página de pré-visualização
        return render_template('preview.html',
                             metadata=metadata,
                             data=data,
                             metadata_json=metadata_json,
                             data_json=data_json,
                             template_choice=template_choice,
                             ai_enhancement=ai_enhancement)

    except Exception as e:
        return render_template('index.html', error=f"Erro ao processar arquivos: {str(e)}")

@app.route('/generate', methods=['POST'])
def generate_reports():
    """Gera os relatórios com base nos dados fornecidos"""
    try:
        # Obter dados do formulário
        metadata_json = request.form.get('metadata_json')
        data_json = request.form.get('data_json')
        template_choice = request.form.get('template_choice', 'all')
        output_dir = request.form.get('output_dir', OUTPUT_DIR)

        # Converter JSON de volta para objetos Python
        metadata = json.loads(metadata_json)
        data = json.loads(data_json)

        # Gerar relatórios
        filepaths = generate_reports_from_data(metadata, data, template_choice, output_dir)

        # Retornar página de resultados
        return render_template('results.html',
                             filepaths=filepaths,
                             template_choice=template_choice,
                             output_dir=output_dir)

    except Exception as e:
        return render_template('index.html', error=f"Erro ao gerar relatórios: {str(e)}")

@app.route('/download/<path:filename>')
def download_file(filename):
    """Permite download dos arquivos gerados"""
    try:
        # Determinar o caminho completo do arquivo
        file_path = os.path.join(OUTPUT_DIR, filename)

        # Verificar se o arquivo existe
        if not os.path.exists(file_path):
            return render_template('index.html', error="Arquivo não encontrado.")

        # Enviar arquivo para download
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return render_template('index.html', error=f"Erro ao fazer download: {str(e)}")

def main():
    """Função principal para iniciar o servidor web"""
    print("Iniciando AUTO-RELATÓRIOS CIENTÍFICOS - Interface Web...")
    print("Acesse http://localhost:5000 no seu navegador")

    # Criar diretório de saída se não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Iniciar servidor Flask
    app.run(host='localhost', port=5000, debug=True)

if __name__ == '__main__':
    main()