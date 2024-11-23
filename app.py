from flask import Flask, request, jsonify

import os
from flask_cors import CORS
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_core.prompts import PromptTemplate
import pandas as pd

app = Flask(__name__)
CORS(app)


with open("openai.txt", "r") as conf:
    os.environ['OPENAI_API_KEY'] = conf.read().strip()

EXTRACT_PROMPT_TEMPLATE = """
Extract the key information from the following PDF content:

PDF Content:
{pdf_content}

And provide the answer according to the following instruction:

{instruction}
"""

SUMMARIZE_PROMPT_TEMPLATE = """
Summarize information from the following PDF content:

PDF Content:
{pdf_content}
"""

CATEGORIZATION_PROMPT_TEMPLATE = """
Group information into categories from the following PDF content:

PDF Content:
{pdf_content}
"""

STRUCTURE_OPTIMISATION_PROMPT_TEMPLATE = """
Correct grammar, spelling, and punctuation errors, improve structure, 
and flag instances of passive voice, jargon, or repetitive phrases of text from following PDF content:

PDF Content:
{pdf_content}
"""

GENERATION_EMAIL_PROMPT_TEMPLATE = """
According to the following PDF content and User query generate professional email:

PDF Content:
{pdf_content}

User query:
{user_query}
"""

HIGHLIGHT_PROMPT_TEMPLATE = """
Highlight the most important sentences or paragraphs in the information from the following PDF content:

PDF Content:
{pdf_content}
"""

EXTRACT_PROMPT = PromptTemplate(
    input_variables=["pdf_content", "instruction"],
    template=EXTRACT_PROMPT_TEMPLATE
)

SUMMARIZE_PROMPT = PromptTemplate(
    input_variables=["pdf_content"],
    template=SUMMARIZE_PROMPT_TEMPLATE
)

CATEGORIZE_PROMPT = PromptTemplate(
    input_variables=["pdf_content"],
    template=CATEGORIZATION_PROMPT_TEMPLATE
)

STRUCTURE_OPTIMISATION_PROMPT = PromptTemplate(
    input_variables=["pdf_content"],
    template=SUMMARIZE_PROMPT_TEMPLATE
)

GENERATION_EMAIL_PROMPT = PromptTemplate(
    input_variables=["pdf_content"],
    template=GENERATION_EMAIL_PROMPT_TEMPLATE
)

HIGHLIGHT_PROMPT = PromptTemplate(
    input_variables=["pdf_content"],
    template=HIGHLIGHT_PROMPT_TEMPLATE
)

MODEL = "gpt-4o"
IDB = "log.csv"

data_path = os.path.join(os.path.dirname(__file__) + '/data/')

@app.route('/', methods=['GET', 'POST'])
def main():
    return jsonify({"code": "200", "message": "DocFusion AI is working. Use /api/v1/analyze endpoint to work with your documents."}), 200


def chain_extract(model, content, instruction):
    llm = ChatOpenAI(model_name=model, streaming=False, temperature=0.7)
    extract_chain = LLMChain(llm=llm, prompt=EXTRACT_PROMPT)
    extracted_info = extract_chain.run({"pdf_content": content, "instruction": instruction})
    log_extracted_info = {
        'timestamp': pd.Timestamp.now(),
        'type': 'answer',
        'data': extracted_info
    }

    data_frame_extracted_info = pd.DataFrame([log_extracted_info])

    if os.path.exists(IDB):
        data_frame_extracted_info.to_csv(IDB, mode='a', header=False, index=False)

    return extracted_info


def process_file(file_name, file_type):
    try:
        f = request.files['file']
        f.save(os.path.join(data_path + file_name + "." + file_type))
    except FileNotFoundError:
        os.makedirs(os.path.join(data_path))
        f = request.files['file']
        f.save(os.path.join(data_path + file_name + "." + file_type))

    file_path = os.path.join(data_path + file_name + "." + file_type)

    if not os.path.exists(file_path):
        return jsonify({"code":"404", "message": "Error while uploading file. File might be corrupted or your connection was abnormally aborted."}), 500

    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = UnstructuredWordDocumentLoader(file_path)

    document = loader.load()
    content = [doc.page_content for doc in document]
    return content


@app.route('/api/v1/analyze', methods=['GET', 'POST'])
def analyze_file():
    file_name = request.form['filename']
    file_type = request.form['filetype']
    prompt = request.form['prompt']

    if not file_name:
        return jsonify({"code":"400", "message": "Please provide a file name."}), 400

    if not prompt:
        return jsonify({"code":"400", "message": "Please provide a prompt."}), 400

    file = process_file(file_name, file_type)
    output =  chain_extract(MODEL, file, prompt)

    return jsonify({"code": "200", "message": output}), 200

def chain_summarize(model, content):
    llm = ChatOpenAI(model_name=model, streaming=False, temperature=0.7)
    extract_chain = LLMChain(llm=llm, prompt=SUMMARIZE_PROMPT)
    extracted_info = extract_chain.run({"pdf_content": content})
    log_extracted_info = {
        'timestamp': pd.Timestamp.now(),
        'type': 'answer',
        'data': extracted_info
    }

    data_frame_extracted_info = pd.DataFrame([log_extracted_info])

    if os.path.exists(IDB):
        data_frame_extracted_info.to_csv(IDB, mode='a', header=False, index=False)

    return extracted_info

@app.route('/api/v1/summarize', methods=['GET', 'POST'])
def summarize_file():
    file_name = request.form['filename']
    file_type = request.form['filetype']

    if not file_name:
        return jsonify({"code":"400", "message": "Please provide a file name."}), 400

    file = process_file(file_name, file_type)
    output =  chain_summarize(MODEL, file)

    return jsonify({"code": "200", "message": output}), 200

def chain_categorize(model, content):
    llm = ChatOpenAI(model_name=model, streaming=False, temperature=0.7)
    extract_chain = LLMChain(llm=llm, prompt=CATEGORIZE_PROMPT)
    extracted_info = extract_chain.run({"pdf_content": content})
    log_extracted_info = {
        'timestamp': pd.Timestamp.now(),
        'type': 'answer',
        'data': extracted_info
    }

    data_frame_extracted_info = pd.DataFrame([log_extracted_info])

    if os.path.exists(IDB):
        data_frame_extracted_info.to_csv(IDB, mode='a', header=False, index=False)

    return extracted_info

@app.route('/api/v1/categorize', methods=['GET', 'POST'])
def categorize_file():
    file_name = request.form['filename']
    file_type = request.form['filetype']

    if not file_name:
        return jsonify({"code":"400", "message": "Please provide a file name."}), 400

    file = process_file(file_name, file_type)
    output =  chain_categorize(MODEL, file)

    return jsonify({"code": "200", "message": output}), 200

def chain_optimize(model, content):
    llm = ChatOpenAI(model_name=model, streaming=False, temperature=0.7)
    extract_chain = LLMChain(llm=llm, prompt=STRUCTURE_OPTIMISATION_PROMPT)
    extracted_info = extract_chain.run({"pdf_content": content})
    log_extracted_info = {
        'timestamp': pd.Timestamp.now(),
        'type': 'answer',
        'data': extracted_info
    }

    data_frame_extracted_info = pd.DataFrame([log_extracted_info])

    if os.path.exists(IDB):
        data_frame_extracted_info.to_csv(IDB, mode='a', header=False, index=False)

    return extracted_info

@app.route('/api/v1/optimize', methods=['GET', 'POST'])
def optimize_file_structure():
    file_name = request.form['filename']
    file_type = request.form['filetype']

    if not file_name:
        return jsonify({"code":"400", "message": "Please provide a file name."}), 400

    file = process_file(file_name, file_type)
    output =  chain_optimize(MODEL, file)

    return jsonify({"code": "200", "message": output}), 200

def chain_generate_email(model, content, user_query):
    llm = ChatOpenAI(model_name=model, streaming=False, temperature=0.7)
    extract_chain = LLMChain(llm=llm, prompt=GENERATION_EMAIL_PROMPT)
    extracted_info = extract_chain.run({"pdf_content": content, "user_query": user_query})
    log_extracted_info = {
        'timestamp': pd.Timestamp.now(),
        'type': 'answer',
        'data': extracted_info
    }

    data_frame_extracted_info = pd.DataFrame([log_extracted_info])

    if os.path.exists(IDB):
        data_frame_extracted_info.to_csv(IDB, mode='a', header=False, index=False)

    return extracted_info

@app.route('/api/v1/email', methods=['GET', 'POST'])
def generate_email():
    file_name = request.form['filename']
    file_type = request.form['filetype']
    query = request.form['query']

    if not file_name:
        return jsonify({"code":"400", "message": "Please provide a file name."}), 400

    file = process_file(file_name, file_type)
    output =  chain_generate_email(MODEL, file, query)

    return jsonify({"code": "200", "message": output}), 200

def chain_highlight(model, content):
    llm = ChatOpenAI(model_name=model, streaming=False, temperature=0.7)
    extract_chain = LLMChain(llm=llm, prompt=HIGHLIGHT_PROMPT)
    extracted_info = extract_chain.run({"pdf_content": content})
    log_extracted_info = {
        'timestamp': pd.Timestamp.now(),
        'type': 'answer',
        'data': extracted_info
    }

    data_frame_extracted_info = pd.DataFrame([log_extracted_info])

    if os.path.exists(IDB):
        data_frame_extracted_info.to_csv(IDB, mode='a', header=False, index=False)

    return extracted_info

@app.route('/api/v1/highlight', methods=['GET', 'POST'])
def highlight_file():
    file_name = request.form['filename']
    file_type = request.form['filetype']

    if not file_name:
        return jsonify({"code": "400", "message": "Please provide a file name."}), 400

    file = process_file(file_name, file_type)
    output = chain_highlight(MODEL, file)

    return jsonify({"code": "200", "message": output}), 200

if __name__ == '__main__':
    app.run()
