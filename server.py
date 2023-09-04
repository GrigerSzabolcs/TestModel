import numpy as np
from flask import Flask, request, jsonify
import keras
import tensorflow_addons as tfa
from keras_preprocessing.text import tokenizer_from_json
import json
from keras_preprocessing.sequence import pad_sequences


with open("tokenizer.json") as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

app = Flask(__name__)

#model = keras.models.load_model("models/CNN_model.h5", custom_objects={"F1Score": tfa.metrics.F1Score(average='macro',num_classes=3)})
model = keras.models.load_model("CNN_model.h5")

#bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#model = keras.models.load_model("models/BERT_model.h5", custom_objects={"TFBertModel": TFBertModel})



@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)

    #input_ids = []
    #attention_masks = []
    #for sent in tqdm(data['text']):
    #    bert_inp = bert_tokenizer.encode_plus(sent, add_special_tokens=True, max_length=128, pad_to_max_length=True,
    #                                          return_attention_mask=True)
    #    input_ids.append(bert_inp['input_ids'])
    #    attention_masks.append(bert_inp['attention_mask'])

    #input_ids = np.asarray(input_ids)
    #attention_masks = np.array(attention_masks)

    X_text = np.array(data['text'])
    X_data = tokenizer.texts_to_sequences(X_text)
    maxlength=30
    X_data = pad_sequences(X_data, padding='post', maxlen=maxlength)
    #prediction = model.predict([input_ids, attention_masks])
    prediction = model.predict(X_data)

    answers = []
    for x in prediction:
        if x[0] > x[1] and x[0] > x[2]:
            answers.append("negative")
        elif x[1] > x[0] and x[1] > x[2]:
            answers.append("neutral")
        else:
            answers.append("positive")
    # 1 means neutral, 0 means negative, 2 means positive
    """
    result = ""
    if prediction[0][0] > prediction[0][1] and prediction[0][0] > prediction[0][2]:
        result = "negative"
    elif prediction[0][1] > prediction[0][0] and prediction[0][1] > prediction[0][2]:
        result = "neutral"
    else:
        result = "positive"
    """
    print('asd')
    return jsonify(answers)

if __name__ == '__main__':
    app.run(port=5000, debug=True)