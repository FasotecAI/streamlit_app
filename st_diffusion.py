import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


file_path ="fasotec_ai.jpg"
img = Image.open(file_path)
model_id = "stabilityai/stable-diffusion-2-1"
 # meta-llama/CodeLlama-7b-hf

def main(img):

# sidebar
    st.sidebar.image(img)
 
    options = ["stabilityai/stable-diffusion-2-1", "stabilityai/stable-diffusion-3-medium"]
    choice = st.sidebar.selectbox("Select an model name", options)
    
    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    st.title('生成AI(Text2Img): StableDiffusion')
   
    col1, col2 = st.columns(2)
    col1.header("Prompt")
    col2.header("生成画像")
    user_input = col1.text_input("Enter your Prompt(英語)")
    if st.button("Generate Image"):
        if user_input:
            with st.spinner("Generating Image ..."): 
                prompt = user_input
                image = pipe(prompt).images[0]
                #out_file = "output.png"
                #image.save(out_file)
                
                col2.image(image, use_column_width=True) #768*768
            st.success('Done!')    
        else:
            st.write("Promptを入力してください。")

if __name__ == "__main__":
    main(img)
