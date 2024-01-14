import streamlit as st
import requests
from requests_toolbelt import MultipartEncoder
from PIL import Image
import io
import base64
import pandas as pd
import st_aggrid


css='''
<style>
    section.main > div {max-width:60rem}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

# Pada halaman Map, pengguna dapat melakukan visualisasi hasil segmentasi tutupan lahan pada dataset [LandCoverAI](https://landcover.ai.linuxpolska.com/),
#     yang diambil dari Polandia, berdasarkan provinsi. Pengguna dapat berinteraksi dengan map, dan melihat area tutupan lahan per kelas. Gambar segmentasi dibuat menggunakan model UNetFormer++.
st.sidebar.header("Page Instructions")
st.sidebar.info(
    """
    
    On the Map page, users can visualize the results of land cover segmentation on the [LandCoverAI](https://landcover.ai.linuxpolska.com/) dataset, taken from Poland, by province. The user can interact with the map, and view the land cover area per class. The segmentation image is created using the UNetFormer++ model.
    """
)

def fetch_all_area_summary():
    response = requests.get("http://localhost:8000/fetch_mask_pred_area_all_data")
    data = response.json()
    if response.status_code == 200:
        return data
    else:
        return []

def fetch_mask_pred_data(prov_name:str):

    m = MultipartEncoder(
        fields={
            'prov_filter': prov_name
        }
    )

    response = requests.post("http://localhost:8000/fetch_mask_pred_data",
                      data=m,
                      headers={'Content-Type': m.content_type},
                      timeout=8000
                    )

    # response = requests.get("http://localhost:8000/fetch_mask_pred_data")
    data = response.json()
    if response.status_code == 200:
        return data
    else:
        return []


def fetch_map():
    response = requests.get("http://localhost:8000/map")
    
    if response.status_code == 200:
        # html_content = response.text
        return response.text
    else:
        return None

def get_prov_map(prov_filter: str):
    m = MultipartEncoder(
        fields={
            'prov_filter': prov_filter
        }
    )

    r = requests.post("http://localhost:8000/map",
                      data=m,
                      headers={'Content-Type': m.content_type},
                      timeout=8000
                    )
    
    if r.status_code == 200:
        # html_content = response.text
        return r.text
    else:
        return 

def show_all_summary_data(data):
    if data:
        with st.container():
            table_data = {
                "Total Area":f"{data['area']} m\u00B2",
                "Total Building Area":f"{data['area']*data['Building_pct']:.2f} m\u00B2 ({data['Building_pct']*100:.2f} %)",
                "Total Woodland Area":f"{data['area']*data['Woodland_pct']:.2f} m\u00B2 ({data['Woodland_pct']*100:.2f} %)",
                "Total Water Area":f"{data['area']*data['Water_pct']:.2f} m\u00B2 ({data['Water_pct']*100:.2f} %)",
                "Total Road Area":f"{data['area']*data['Road_pct']:.2f}m\u00B2 ({data['Road_pct']*100:.2f} %)",

            }
            df = pd.DataFrame(list(table_data.items()), columns=['Key', 'Value'])

            MIN_HEIGHT = 100
            MAX_HEIGHT = 180
            ROW_HEIGHT = 50
            
            # st.dataframe(df, hide_index=True, use_container_width=True)
            st_aggrid.AgGrid(df,fit_columns_on_grid_load=True, height=min(MIN_HEIGHT + len(df) * ROW_HEIGHT, MAX_HEIGHT))
        st.write("\n")

def show_data(data):
    if data:
        st.subheader("Provinces Area Segmentation Summary")
        # print(data)
        
        for item in data:
            img_path_base64 = item["img_path"]

            img_path_bytes = base64.b64decode(img_path_base64.encode())
            img_buf = io.BytesIO(img_path_bytes)

            img_path = Image.open(img_buf)

            with st.container():
                st.write(f"**{item['img_id']}** in **{item['province']}**")
                # st.subheader(f"**{item['timestamp'][0]}** at **{item['timestamp'][1]}** using **{item['model']}**")
                col1, col2 = st.columns(2)
                # with col1:
                # image_comparison(
                #     img1=img_path,
                #     img2=output_path,
                #     label1="Original",
                #     label2="Predicted",
                # )
                with col1:
                    with st.container():
                        st.image(img_path)
                with col2:
                    with st.container():
                        # st.write(f"Total Area: {item['area']}")
                        # st.write(f"Total Building Area: {item['area']*item['Building_pct']}")
                        # st.write(f"Total Woodland Area: {item['area']*item['Woodland_pct']}")
                        # st.write(f"Total Water Area: {item['area']*item['Water_pct']}")
                        # st.write(f"Total Road Area: {item['area']*item['Road_pct']}")
                        
                        table_data = {
                            "Total Area":f"{item['area']} m\u00B2",
                            "Total Building Area":f"{item['area']*item['Building_pct']:.2f} m\u00B2 ({item['Building_pct']*100:.2f} %)",
                            "Total Woodland Area":f"{item['area']*item['Woodland_pct']:.2f} m\u00B2 ({item['Woodland_pct']*100:.2f} %)",
                            "Total Water Area":f"{item['area']*item['Water_pct']:.2f} m\u00B2 ({item['Water_pct']*100:.2f} %)",
                            "Total Road Area":f"{item['area']*item['Road_pct']:.2f}m\u00B2 ({item['Road_pct']*100:.2f} %)",

                        }
                        df = pd.DataFrame(list(table_data.items()), columns=['Key', 'Value'])
                        
                        # MIN_HEIGHT = 100
                        # MAX_HEIGHT = 170
                        # ROW_HEIGHT = 50
                        
                        # st.dataframe(df, hide_index=True, use_container_width=True)
                        st_aggrid.AgGrid(df,fit_columns_on_grid_load=True)
                # break
            st.write("\n")

st.title("Map Visualization")
# st.markdown("***")


prov_filter = st.selectbox(
    label="Select a province to filter by: ",
    options=(
        'Select a Province',
        'all',
        'śląskie',
        'opolskie',
        'wielkopolskie',
        'zachodniopomorskie',
        'świętokrzyskie',
        'kujawsko-pomorskie',
        'podlaskie',
        'dolnośląskie',
        'podkarpackie',
        'małopolskie',
        'pomorskie',
        'warmińsko-mazurskie',
        'łódzkie',
        'mazowieckie',
        'lubelskie',
        'lubuskie'
    ),
    # index=None
)

# m = folium.Map(location=[39.949610, -75.150282], zoom_start=16)
# FloatImage('leg.png',bottom=1, left=1, width='150px').add_to(m)
# st_data = st_folium(m, width=725)



with st.spinner("Loading Map"):
    if prov_filter != 'Select a Province':
        if prov_filter == 'all':
            if f'html_map_all' not in st.session_state:
                html_content = get_prov_map(prov_filter)
                if html_content is not None:
                    st.session_state[f'html_map_all'] = html_content
                    st.components.v1.html(html_content,height=550)
                    # st.markdown(html_content, unsafe_allow_html=True)
                else:
                    st.write("Error")
            else:
                st.components.v1.html(st.session_state[f'html_map_all'] ,height=550)
                # st.markdown(st.session_state[f'html_map_all'], unsafe_allow_html=True)
        else:
            html_content = get_prov_map(prov_filter)
            if html_content is not None:
                st.components.v1.html(html_content,height=550)
                # st.markdown(html_content, unsafe_allow_html=True)
            else:
                st.write("Error")

st.write("\n")

with st.spinner("Loading Images"):
    if prov_filter != 'Select a Province':
        if prov_filter == 'all':
            if f'images_all' not in st.session_state:
                st.subheader("All Area Segmentation Summary")
                all_data_summary = fetch_all_area_summary()
                show_all_summary_data(all_data_summary)

                data = fetch_mask_pred_data(prov_filter)
                show_data(data)
                st.session_state[f'images_all'] = data
            else:
                st.subheader("All Area Segmentation Summary")
                all_data_summary = fetch_all_area_summary()
                show_all_summary_data(all_data_summary)
                show_data(st.session_state[f'images_all'])

        else:
            data = fetch_mask_pred_data(prov_filter)
            show_data(data)