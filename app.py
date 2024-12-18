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

TRANSLATE_PROMPT_TEMPLATE = """
Translate the following information to the language specified:

PDF Content:
{pdf_content}

Translate to:
{language}

Write only translated text and nothing else.
"""

SUMMARIZE_PROMPT_TEMPLATE = """
Provide an answer according to following Instruction, 
summarize information from the following PDF content:

PDF Content:
{pdf_content}

Instruction:
{instruction}
"""

CATEGORIZATION_PROMPT_TEMPLATE = """
Provide an answer according to following Instruction, 
group information into categories from the following PDF content:

PDF Content:
{pdf_content}

Instruction:
{instruction}
"""

STRUCTURE_OPTIMISATION_PROMPT_TEMPLATE = """
Provide an answer according to following Instruction, 
correct grammar, spelling, and punctuation errors, improve structure, 
and flag instances of passive voice, jargon, or repetitive 
phrases of text from following PDF content:

PDF Content:
{pdf_content}

Instruction:
{instruction}
"""

GENERATION_EMAIL_PROMPT_TEMPLATE = """
According to the following PDF content and User query generate professional email:

PDF Content:
{pdf_content}

User query:
{user_query}

And provide the answer according to the following instruction:
{instruction}
"""

HIGHLIGHT_PROMPT_TEMPLATE = """
provide an answer according to following Instruction,
highlight the most important sentences or paragraphs 
in the information from the following PDF content:

PDF Content:
{pdf_content}

Instruction:
{instruction}
"""

VALIDATION_PROMPT_TEMPLATE = """
Examine the text for errors in grammar, syntax, and structure. 
Return the revised text if changes are needed; if perfect, simply output the same text, and don`t write any
messages:

Text:
{text}
"""

EXTRACT_PROMPT = PromptTemplate(
    input_variables=["pdf_content", "instruction"],
    template=EXTRACT_PROMPT_TEMPLATE
)

TRANSLATE_PROMPT = PromptTemplate(
    input_variables=["pdf_content", "language"],
    template=TRANSLATE_PROMPT_TEMPLATE
)

SUMMARIZE_PROMPT = PromptTemplate(
    input_variables=["pdf_content", "instruction"],
    template=SUMMARIZE_PROMPT_TEMPLATE
)

CATEGORIZE_PROMPT = PromptTemplate(
    input_variables=["pdf_content", "instruction"],
    template=CATEGORIZATION_PROMPT_TEMPLATE
)

STRUCTURE_OPTIMISATION_PROMPT = PromptTemplate(
    input_variables=["pdf_content", "instruction"],
    template=STRUCTURE_OPTIMISATION_PROMPT_TEMPLATE
)

GENERATION_EMAIL_PROMPT = PromptTemplate(
    input_variables=["pdf_content", "instruction"],
    template=GENERATION_EMAIL_PROMPT_TEMPLATE
)

HIGHLIGHT_PROMPT = PromptTemplate(
    input_variables=["pdf_content", "instruction"],
    template=HIGHLIGHT_PROMPT_TEMPLATE
)

VALIDATE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=VALIDATION_PROMPT_TEMPLATE
)

MODEL = "gpt-4o"
IDB = "log.csv"

data_path = os.path.join(os.path.dirname(__file__) + '/data/')
@app.route('/', methods=['GET', 'POST'])
def main():
    return jsonify({"code": "200", "message": "DocFusion AI is working. Use /api/v1/analyze endpoint to work with your documents."}), 200

def chain_validate(model, text):
    llm = ChatOpenAI(model_name=model, streaming=False, temperature=0.7)

    validate_chain = LLMChain(llm=llm, prompt=VALIDATE_PROMPT)
    validated_info = validate_chain.run({"text": text})

    log_extracted_info = {
        'timestamp': pd.Timestamp.now(),
        'type': 'answer',
        'data': validated_info
    }

    data_frame_extracted_info = pd.DataFrame([log_extracted_info])

    if os.path.exists(IDB):
        data_frame_extracted_info.to_csv(IDB, mode='a', header=False, index=False)

    return validated_info

def chain_extract(model, content, instruction):
    llm = ChatOpenAI(model_name=model, streaming=False, temperature=0.7)
    extract_chain = LLMChain(llm=llm, prompt=EXTRACT_PROMPT)

    extracted_info = chain_validate(MODEL, extract_chain.run({"pdf_content": content,
                                                              "instruction": instruction}))

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
    prompt = request.form['query']

    if not file_name:
        return jsonify({"code":"400", "message": "Please provide a file name."}), 400

    if not prompt:
        return jsonify({"code":"400", "message": "Please provide a prompt."}), 400

    file = process_file(file_name, file_type)
    output =  chain_extract(MODEL, file, prompt)

    return jsonify({"code": "200", "message": output}), 200

def chain_translate(model, content, language):
    llm = ChatOpenAI(model_name=model, streaming=False, temperature=0.7)
    translate_chain = LLMChain(llm=llm, prompt=TRANSLATE_PROMPT)

    translated_info = chain_validate(MODEL, translate_chain.run({"pdf_content": content, "language": language}))

    log_extracted_info = {
        'timestamp': pd.Timestamp.now(),
        'type': 'answer',
        'data': translated_info
    }

    data_frame_extracted_info = pd.DataFrame([log_extracted_info])

    if os.path.exists(IDB):
        data_frame_extracted_info.to_csv(IDB, mode='a', header=False, index=False)

    return translated_info

@app.route('/api/v1/translate', methods=['GET', 'POST'])
def translate_file():
    file_name = request.form['filename']
    file_type = request.form['filetype']
    lang = request.form['query']

    if not file_name:
        return jsonify({"code":"400", "message": "Please provide a file name."}), 400

    if not lang:
        return jsonify({"code":"400", "message": "Please provide a language."}), 400

    file = process_file(file_name, file_type)
    output = chain_translate(MODEL, file, lang)

    return jsonify({"code": "200", "message": output}), 200

def chain_summarize(model, content, instruction):
    llm = ChatOpenAI(model_name=model, streaming=False, temperature=0.7)
    summarize_chain = LLMChain(llm=llm, prompt=SUMMARIZE_PROMPT)

    summarized_info = chain_validate(MODEL, summarize_chain.run({"pdf_content": content,
                                                                 "instruction": instruction}))

    log_extracted_info = {
        'timestamp': pd.Timestamp.now(),
        'type': 'answer',
        'data': summarized_info
    }

    data_frame_extracted_info = pd.DataFrame([log_extracted_info])

    if os.path.exists(IDB):
        data_frame_extracted_info.to_csv(IDB, mode='a', header=False, index=False)

    return summarized_info

@app.route('/api/v1/summarize', methods=['GET', 'POST'])
def summarize_file():
    file_name = request.form['filename']
    file_type = request.form['filetype']
    instruction = request.form['query']

    if not file_name:
        return jsonify({"code":"400", "message": "Please provide a file name."}), 400

    file = process_file(file_name, file_type)
    output =  chain_summarize(MODEL, file, instruction)

    return jsonify({"code": "200", "message": output}), 200

def chain_categorize(model, content, instruction):
    llm = ChatOpenAI(model_name=model, streaming=False, temperature=0.7)
    categorize_chain = LLMChain(llm=llm, prompt=CATEGORIZE_PROMPT)

    categorized_info = chain_validate(MODEL, categorize_chain.run({"pdf_content": content,
                                                                   "instruction": instruction}))

    log_extracted_info = {
        'timestamp': pd.Timestamp.now(),
        'type': 'answer',
        'data': categorized_info
    }

    data_frame_extracted_info = pd.DataFrame([log_extracted_info])

    if os.path.exists(IDB):
        data_frame_extracted_info.to_csv(IDB, mode='a', header=False, index=False)

    return categorized_info

@app.route('/api/v1/categorize', methods=['GET', 'POST'])
def categorize_file():
    file_name = request.form['filename']
    file_type = request.form['filetype']
    instruction = request.form['query']

    if not file_name:
        return jsonify({"code":"400", "message": "Please provide a file name."}), 400

    file = process_file(file_name, file_type)
    output =  chain_categorize(MODEL, file, instruction)


    return jsonify({"code": "200", "message": output}), 200

def chain_optimize(model, content, instruction):
    llm = ChatOpenAI(model_name=model, streaming=False, temperature=0.7)
    optimize_chain = LLMChain(llm=llm, prompt=STRUCTURE_OPTIMISATION_PROMPT)

    optimized_info = chain_validate(MODEL, optimize_chain.run({"pdf_content": content,
                                                               "instruction": instruction}))

    log_extracted_info = {
        'timestamp': pd.Timestamp.now(),
        'type': 'answer',
        'data': optimized_info
    }

    data_frame_extracted_info = pd.DataFrame([log_extracted_info])

    if os.path.exists(IDB):
        data_frame_extracted_info.to_csv(IDB, mode='a', header=False, index=False)

    return optimized_info

@app.route('/api/v1/optimize', methods=['GET', 'POST'])
def optimize_file_structure():
    file_name = request.form['filename']
    file_type = request.form['filetype']
    instruction = request.form['query']

    if not file_name:
        return jsonify({"code":"400", "message": "Please provide a file name."}), 400

    file = process_file(file_name, file_type)
    output =  chain_optimize(MODEL, file, instruction)

    return jsonify({"code": "200", "message": output}), 200

def chain_generate_email(model, content, user_query):
    llm = ChatOpenAI(model_name=model, streaming=False, temperature=0.7)
    generate_chain = LLMChain(llm=llm, prompt=GENERATION_EMAIL_PROMPT)

    generated_email = chain_validate(MODEL, generate_chain.run({"pdf_content": content,
                                                              "user_query": user_query}))

    log_extracted_info = {
        'timestamp': pd.Timestamp.now(),
        'type': 'answer',
        'data': generated_email
    }

    data_frame_extracted_info = pd.DataFrame([log_extracted_info])

    if os.path.exists(IDB):
        data_frame_extracted_info.to_csv(IDB, mode='a', header=False, index=False)

    return generated_email

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

def chain_highlight(model, content, instruction):
    llm = ChatOpenAI(model_name=model, streaming=False, temperature=0.7)
    highlight_chain = LLMChain(llm=llm, prompt=HIGHLIGHT_PROMPT)

    highlighted_info = chain_validate(MODEL, highlight_chain.run({"pdf_content": content,
                                                                  "instruction": instruction}))

    log_extracted_info = {
        'timestamp': pd.Timestamp.now(),
        'type': 'answer',
        'data': highlighted_info
    }

    data_frame_extracted_info = pd.DataFrame([log_extracted_info])

    if os.path.exists(IDB):
        data_frame_extracted_info.to_csv(IDB, mode='a', header=False, index=False)

    return highlighted_info

@app.route('/api/v1/highlight', methods=['GET', 'POST'])
def highlight_file():
    file_name = request.form['filename']
    file_type = request.form['filetype']
    instruction = request.form['query']

    if not file_name:
        return jsonify({"code": "400", "message": "Please provide a file name."}), 400

    file = process_file(file_name, file_type)
    output = chain_highlight(MODEL, file, instruction)

    return jsonify({"code": "200", "message": output}), 200

if __name__ == '__main__':
    app.run()
