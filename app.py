from flask import Flask, request, jsonify

import os
from flask_cors import CORS
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)
CORS(app)

with open("openai.txt", "r") as f:
    os.environ['OPENAI_API_KEY'] = f.read().strip()

EXTRACT_PROMPT_TEMPLATE = """
Extract the key information from the following PDF content:

PDF Content:
{pdf_content}

And provide the answer according to the following instruction:

{instruction}
"""

EXTRACT_PROMPT = PromptTemplate(
    input_variables=["pdf_content", "instruction"],
    template=EXTRACT_PROMPT_TEMPLATE
)

MODEL = "gpt-4o"

data_path = os.path.join(os.path.dirname(__file__) + '/data/')

@app.route('/', methods=['GET', 'POST'])
def main():
    return jsonify({"code": "200", "message": "DocFusion AI is working. Use /api/v1/analyze endpoint to work with your documents."}), 200


@app.route('/api/v1/analyze', methods=['GET', 'POST'])
def analyze_file():
    file_name = request.form['filename']
    prompt = request.form['prompt']

    if not file_name:
        return jsonify({"code":"400", "message": "Please provide a file name."}), 400

    if not prompt:
        return jsonify({"code":"400", "message": "Please provide a prompt."}), 400

    try:
        f = request.files['file']
        f.save(os.path.join(data_path + file_name + ".pdf"))
    except FileNotFoundError:
        os.makedirs(os.path.join(data_path))
        f = request.files['file']
        f.save(os.path.join(data_path + file_name + ".pdf"))

    file_path = os.path.join(data_path + file_name + ".pdf")
    if not os.path.exists(file_path):
        return jsonify({"code":"404", "message": "Error while uploading file. File might be corrupted or your connection was abnormally aborted."}), 500

    loader = PyPDFLoader(file_path)
    document = loader.load()
    content = [doc.page_content for doc in document]

    llm = ChatOpenAI(model_name=MODEL, streaming=False, temperature=0.7)
    extract_chain = LLMChain(llm=llm, prompt=EXTRACT_PROMPT)
    extracted_info = extract_chain.run(pdf_content=str(content), instruction=prompt)

    return jsonify({"code":"200", "message": extracted_info}), 200


if __name__ == '__main__':
    app.run()
