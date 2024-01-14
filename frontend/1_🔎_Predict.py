import streamlit as st
from requests_toolbelt import MultipartEncoder
import requests
from PIL import Image
import io
import base64
import pandas as pd
import plotly.express as px
from streamlit_image_comparison import image_comparison


# Pada halaman Predict, pengguna dapat mengunggah gambar citra udara untuk melakukan segmentasi tutupan lahan menggunakan beberapa model yang telah dilatih dalam penelitian ini.

st.sidebar.header("Page Instructions")
st.sidebar.info(
    """
    On the Predict page, users can upload aerial images to perform land cover segmentation using several models that have been trained in this research.
    """
)

    # Model dilatih menggunakan gambar citra udara dengan resolusi spasial sebesar **25cm** dan **50cm**. Oleh karena itu, model akan berfungsi dengan baik pada gambar citra udara yang memiliki resolusi spasial dalam rentang tersebut.

st.sidebar.header("Note")
st.sidebar.info(
    """
    The model was trained using aerial imagery with spatial resolutions of **25cm** and **50cm**. Therefore, the model will function well on aerial imagery that has a spatial resolution within this range.
    """
)
# css='''
# <style>
#     section.main > div {max-width:75rem}
# </style>
# '''
# st.markdown(css, unsafe_allow_html=True)

# st.set_page_config(layout="wide")


url = "http://localhost:8000"
endpoint = '/segmentation'

# @st.cache(allow_output_mutation=True)
# @st.cache_data
def process(image, server_url: str, model_choice: str):

    m = MultipartEncoder(
        fields={
            'file': ('filename', image, 'image/jpeg'),
            'model_choice': model_choice
        }
    )

    r = requests.post(server_url,
                      data=m,
                      headers={'Content-Type': m.content_type},
                      timeout=8000
                    )
    # image_bytes = io.BytesIO()
    # files = {'file': ("image.jpg", image_bytes.getvalue())}
    # files = {'file': ("image.jpg", image_bytes.getvalue())}
    # data = {'model_choice': model_choice}
    # r = requests.post(server_url, files=image.getvalue(), params=data)

    return r



st.title("Aerial Imagery Semantic Segmentation")

st.markdown("***")
model_choice = st.radio("Choose a model",
                        ["UNet", "UNet++", "DeepLabV3+", "UNetFormer", "UNetFormer++"]
                    )

st.write("")

file = st.file_uploader("Upload file", type=['jpg','jpeg','png'])
  
show_file = st.empty()

if not file:
    show_file.info(f"Please Upload a file: {', '.join(['jpg','jpeg','png'])}")

if isinstance(file, io.BytesIO):
    # show_file.image(file)

    # segments = process(file, url+endpoint)
    # segmented_image = Image.open(BytesIO(segments.content)).convert('RGB')
    # st.image(segmented_image)

    show_file.image(file)
    if st.button("Start Prediction"):

        with st.spinner("Processing Image"):
            segments = process(file, url + endpoint, model_choice)

        # print(segments)
            result_data = segments.json()
        

            # Decode the base64 image bytes and display them
            if 'ori_img' in result_data:
                ori_img = Image.open(io.BytesIO(base64.b64decode(result_data['ori_img'])))
                # st.image(ori_img, caption="Original Image", use_column_width=True)

            if 'output' in result_data:
                out_img_buff = io.BytesIO(base64.b64decode(result_data['output']))
                output_img = Image.open(out_img_buff)
                # st.image(output_img, caption="Segmented Image", use_column_width=True)

            # with st.container():

            with st.container():
                st.subheader("Prediction Result")
                
                image_comparison(
                    img1=ori_img,
                    img2=output_img,
                    label1="Original",
                    label2="Predicted",
                )

            with st.container():
                st.subheader("Pixel Distribution")
                
                if 'class_counts' in result_data:
                    counts = result_data['class_counts']
                    # df = pd.DataFrame.from_records(counts)
                    # df = pd.read_json(counts, orient='records')
                    
                    # fig = px.bar(df, x='percentage', y='class', color='class', orientation='h',
                    #                 color_discrete_sequence=["black", "red", "green", "blue", "yellow"],
                    #                     category_orders={"class": ["Background", "Building", "Woodland", "Water", "Road"]}
                    #                 )
                    
                    # st.plotly_chart(fig,use_container_width=False)

                    df = pd.read_json(counts, orient='records')
                    df = df.drop(0)
                    df['class_pct_exclude_background'] = df['pixel_count'] / df['pixel_count'].sum()
                    
                    fig = px.bar(df, x='percentage', y='class', color='class', orientation='h',
                                    color_discrete_sequence=["red", "green", "blue", "yellow"],
                                        category_orders={"class": ["Building", "Woodland", "Water", "Road"]}
                                    )
                    
                    st.plotly_chart(fig,use_container_width=False)

            with st.container():

                st.download_button(
                            label="Download Prediction Image",
                            data=out_img_buff,
                            mime="image/jpeg",
                            
                        )
                

    # Reset state untuk melakukan query ulang karena data sudah update
    st.session_state['query_data'] = []