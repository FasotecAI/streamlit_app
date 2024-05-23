import streamlit as st
from transformers import pipeline, AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from PIL import Image

torch.set_default_tensor_type(torch.cuda.FloatTensor)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# モデルとトークナイザーのロード
model_name =  "Salesforce/codegen-350M-mono"  # "bigcode/santacoder" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name) 

# パイプラインの設定#
#chat = pipeline("text-generation", model=model, tokenizer=tokenizer)
file_path ="fasotec_ai.jpg"
img = Image.open(file_path)

def get_response(user_input):
    # モデルを使用して入力に基づく応答を生成
    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    with torch.no_grad():
        tokens = model.generate(input_ids, max_length=256)
    #        **inputs,
    #        max_new_tokens=64,
    #        do_sample=True,
    #        temperature=0.7,
    #        pad_token_id=tokenizer.pad_token_id,
    #    )
    
    #output = tokenizer.decode(tokens[0]) #, skip_special_tokens=True).replace(" ", "")
    output = tokenizer.decode(tokens[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
    return output

def main():

    st.sidebar.image(img)
    options =["Salesforce/codegen-350M-mono", "bigcode/santacoder" ]
    choice = st.sidebar.selectbox("Select a model name", options)


    st.title('生成AIモデル(python関数コード生成)のチャットボット')

    # ユーザー入力フィールド
    user_input = st.text_input("Codeを入力してください: ex) def add(a,b):")

    # 「送信」ボタン
    if st.button("コード生成"):
        if user_input:
            response = get_response(user_input)
            
            st.text_area("コード生成:", value=response, height=500, max_chars=None)
        else:
            st.write("Codeを入力してください。")

if __name__ == "__main__":
    main()
