import streamlit as st
import requests
from streamlit_image_comparison import image_comparison
import base64
from PIL import Image
import io
import plotly.express as px

    # Pada halaman History, pengguna dapat melihat semua segmentasi yang telah dilakukan sebelumnya. Pengguna dapat mengunduh gambar input atau gambar hasil segmentasi, melihat distribusi kelas tutupan lahan, dan melakukan penghapusan history.

st.sidebar.header("Page Instructions")
st.sidebar.info(
    """
    On the History page, users can view all segmentations that have been done previously. Users can download input images or segmentation result images, view land cover class distribution, and perform history deletion.
    """
)

st.title("History")


# Sidebar navigation
# page = st.sidebar.radio("Navigate", ["Home", "History"])

# Function to fetch image paths from the FastAPI server
# def fetch_image_paths():
#     response = requests.get("http://localhost:8000/get_data_img")  # Replace with your FastAPI server URL
#     # print(response)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         return []
    
def fetch_image_data():
    response = requests.get("http://localhost:8000/fetch_image_data")
    data = response.json()
    if response.status_code == 200:
        return data
    else:
        return []
    
def delete_data(img_id):
    delete_url = f"http://localhost:8000/delete_image/{img_id}"  # Replace with your FastAPI server URL
    response = requests.delete(delete_url)

    if response.status_code == 200:
        st.success("Image deleted successfully!")
        # Update the displayed images after deletion
        st.session_state['query_data'] = fetch_image_data()
        st.experimental_rerun()

def delete_callback(btn_id):
    st.session_state[btn_id] = True
    
def show_data(data):
    if data:
        data = data[::-1]
        for item in data:
            img_path_base64 = item["img_path"]
            output_path_base64 = item["output_path"]

            img_path_bytes = base64.b64decode(img_path_base64.encode())
            img_buf = io.BytesIO(img_path_bytes)
            output_path_bytes = base64.b64decode(output_path_base64.encode())
            out_buf = io.BytesIO(output_path_bytes)

            img_path = Image.open(img_buf)
            output_path = Image.open(out_buf)

            with st.container():
                st.write(f"**{item['timestamp'][0]}** at **{item['timestamp'][1]}** using **{item['model']}**")
                # st.subheader(f"**{item['timestamp'][0]}** at **{item['timestamp'][1]}** using **{item['model']}**")
                # col1, col2 = st.columns([3,1])
                # with col1:
                image_comparison(
                    img1=img_path,
                    img2=output_path,
                    label1="Original",
                    label2="Predicted",
                )

                # , on_click=delete_callback, args=[f"delete_{item['img_id']}"]
                # with col1:
                # del_btn_key = f"delete_{item['img_id']}"
                # conf_del_btn_key = f"confirm_delete_{item['img_id']}"
                # del_button = st.button("Delete Data", key=del_btn_key)
                # # confirmation_placeholder = st.empty()
                # if del_button:
                #     st.session_state[del_btn_key] = not st.session_state[del_btn_key]
                    
                # if st.session_state[del_btn_key]:
                #     confirmation = st.button("Confirm Deletion", type='primary', key=conf_del_btn_key)
                #     # if confirmation_placeholder.button("Confirm Deletion", type='primary', key=f"confirm_delete_{item['img_id']}"):
                #     if confirmation:
                #         # st.warning("Are you sure?")
                #         delete_data(item['img_id'])
                #         # Reset state untuk melakukan query ulang karena data sudah update
                #         st.session_state['query_data'] = []
                #         st.rerun()

                class_list = ["Building", "Woodland", "Water", "Road"]
                with st.expander("Pixel Distribution"):
                    pixel_class_count = sum([item[f"{class_count}_count"] for class_count in class_list])
                    df = {
                        "class": class_list,
                        "percentage":[100 * item[f"{class_count}_count"]/pixel_class_count for class_count in class_list],
                    }
                    fig = px.bar(df, x='percentage', y='class', color='class', orientation='h',
                                    color_discrete_sequence=["red", "green", "blue", "yellow"],
                                        category_orders={"class": ["Building", "Woodland", "Water", "Road"]}
                                    )
                    
                    st.plotly_chart(fig,use_container_width=True)

                # class_list = ["Background", "Building", "Woodland", "Water", "Road"]
                # with st.expander("Pixel Distribution"):
                #     df = {
                #         "class": class_list,
                #         "percentage":[item[f"{class_pct}_pct"] for idx, class_pct in enumerate(class_list)],
                #     }
                #     fig = px.bar(df, x='percentage', y='class', color='class', orientation='h',
                #                     color_discrete_sequence=["black", "red", "green", "blue", "yellow"],
                #                         category_orders={"class": ["Background", "Building", "Woodland", "Water", "Road"]}
                #                     )
                    
                #     st.plotly_chart(fig,use_container_width=True)

                with st.expander("Download"):
                    # with col2:
                    col1, col2 = st.columns([1,4])
                    with col1:
                        download_input_key = f"download_input_{item['img_id']}"
                        st.download_button(
                                label="Input Image",
                                data=img_buf,
                                mime="image/jpeg",
                                key=download_input_key,
                            )
                    with col2:
                        download_output_key = f"download_output_{item['img_id']}"
                        st.download_button(
                                label="Prediction Image",
                                data=out_buf,
                                mime="image/jpeg",
                                key=download_output_key,
                            )

                with st.expander("Delete"):
                    st.write("Are you sure?")
                    
                    

                    del_btn_key = f"delete_{item['img_id']}"
                    # del_button = st.button("Delete Data", key=del_btn_key, type="primary")
                    del_button = st.button("Delete Data",  key=del_btn_key, type="primary")
                    if del_button:
                        # st.warning("Are you sure?")
                        delete_data(item['img_id'])
                        # Reset state untuk melakukan query ulang karena data sudah update
                        st.session_state['query_data'] = []
                        # st.session_state['is_expanded'] = False
                        st.rerun()

                        
                

            st.write("\n")
    else:
        st.write("History is empty. Do some prediction first.")

# st.header("History")
# st.write("View past images here.")



with st.spinner("Loading Images"):
    if 'query_data' not in st.session_state or len(st.session_state['query_data']) == 0:
        # st.write("First Case")
        data = fetch_image_data()
        show_data(data)
        
        st.session_state['query_data'] = data
    elif len(st.session_state['query_data']) != 0:
        # st.write("Second Case")
        show_data(st.session_state['query_data'])

