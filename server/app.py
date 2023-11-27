from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pandas as pd
import string
import torch
import transformers
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer

app = Flask(__name__)

# pre-processing
nilai_df = pd.read_csv("nilai_df.csv")
dosen_df = pd.read_csv("dosen_df.csv")
text_cleaned = pd.read_csv("mitra.csv")

cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

# Define the loaded BERT model as a global variable
bert_model = torch.load('bert_model.pth', map_location=torch.device('cpu'))
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', max_length=2048, do_lower_case=True)
label_encoder = torch.load('label_encoder.pth')
        

def predict(question):
    text_enc = tokenizer.encode_plus(
        question,
        None,
        add_special_tokens=True,
        max_length= 512,
        padding = 'max_length',
        return_token_type_ids= False,
        return_attention_mask= True,
        truncation=True,
        return_tensors = 'pt'      
    )
    
    with torch.no_grad():
        outputs = bert_model(text_enc['input_ids'], attention_mask=text_enc['attention_mask'])

    logits = outputs[0]
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    top_10_preds = torch.topk(probabilities, 10)
    top_10_preds = top_10_preds.indices.detach().cpu().numpy().flatten()
    hasil = []
    for i in range(0, 10) :
        hasil.append(text_cleaned['lokasi_mitra'][top_10_preds[i] == text_cleaned['label']].iloc[0])
    decoded_predictions = label_encoder.inverse_transform(top_10_preds.flatten())
    result = []
    for i in range(0, 10) :
        result.append([decoded_predictions[i], hasil[i]])
    return result

def pre_token(doc):
    # case folding
    doc1 = doc.lower()
    # punctuation removal+menghapus angka
    doc2 = doc1.translate(str.maketrans('', '', string.punctuation + string.digits))
    # whitespace removal
    doc3 = doc2.strip()
    return doc3

@app.route("/", methods=["GET", "POST"])
@cross_origin()
def index():
    return 'This is page'

@app.route("/magang", methods=["GET", "POST"])
@cross_origin()
def magangRecommend():
    # get input
    data = request.get_json(force=True)
    nim_input = data['nim']
    minat_input = data['minat']

    # get mk by nim
    mk_query_str = " ".join(nilai_df[nilai_df['nim'].astype(str) == nim_input]['nama_mk'].tolist())

    # get input minat
    bidang_query_str = " ".join(minat_input.split(','))

    student_query = mk_query_str + ' ' + bidang_query_str
    query = pre_token(student_query)

    result = predict(query)
    return jsonify(result)

@app.route("/riset", methods=["GET", "POST"])
@cross_origin()
def risetRecommend():
    # get input
    data = request.get_json(force=True)
    nip_input = data['nip']

    # get data
    judul_str = " ".join((dosen_df[dosen_df['nip'].astype(str) == nip_input]['judul']).tolist())
    abstrak_str = " ".join((dosen_df[dosen_df['nip'].astype(str) == nip_input]['abstrak']).tolist())

    query = judul_str + ' ' + abstrak_str
    cleaned_query = pre_token(query)

    result = predict(cleaned_query)
    return jsonify(result)

#  main thread of execution to start the server
if __name__ == "__main__":
    app.run(debug=True)