from fastapi import FastAPI, File, UploadFile, Depends, Form
from starlette.responses import Response, JSONResponse
import io
from utils.backend_utils import load_model, doPatchify, predict, convert_to_rgb, count_pixel, patch_img, predict_sliding_window
from PIL import Image
import numpy as np
import base64
# from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy import select, column
from datetime import datetime
from uuid import uuid4
import os
# import db_schemas
import db_models
from database import engine, SessionLocal
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import json


db_models.Base.metadata.create_all(bind=engine)



def get_db():
    db = SessionLocal()
    try:
        yield db  # This allows the database session to be used as a context manager
    finally:
        db.close()


app = FastAPI()


@app.post("/segmentation")
def get_segmentation_maps(file: bytes = File(...), model_choice: str = Form(...)):
    model = load_model(model_choice)

    unique_filename = str(uuid4()) + ".jpg"
    storage_img_path = "storage/img/"
    storage_seg_path = "storage/seg/"

    input_image_path = storage_img_path + unique_filename
    segmented_image_path = storage_seg_path + unique_filename
    # segmented_image_path = os.path.join(storage_seg_path, unique_filename)

    ori_img = None
    output = None
    class_counts = None

    # if doPatchify(file):
    #     ori_img, output = patch_img(model, file)
    # else:
    #     ori_img = np.asarray(Image.open(io.BytesIO(file)))
    #     output = predict(model, file)
    #     output = convert_to_rgb(output)

    ori_img = np.asarray(Image.open(io.BytesIO(file)))
    output = predict_sliding_window(model, file)

    class_counts = count_pixel(output)
    # print(class_counts[class_counts['class'] == 'Background']['percentage'].values[0])
    # Convert image data to base64-encoded strings
    ori_img_base64 = None
    output_base64 = None

    if ori_img is not None:
        img_ori = Image.fromarray(ori_img)

        # Save to folder
        img_ori.save(input_image_path, format="JPEG")

        byte_io_ori = io.BytesIO()
        img_ori.save(byte_io_ori, format="JPEG")
        ori_img_bytes = byte_io_ori.getvalue()
        ori_img_base64 = base64.b64encode(ori_img_bytes).decode()

    if output is not None:
        img_out = Image.fromarray(output)

        # Save to folder
        img_out.save(segmented_image_path, format="JPEG")

        byte_io_output = io.BytesIO()
        img_out.save(byte_io_output, format="JPEG")
        output_bytes = byte_io_output.getvalue()
        output_base64 = base64.b64encode(output_bytes).decode()

    class_counts_json = class_counts.to_json(orient="records")
    # print(class_counts_json)
    response_data = {
        "ori_img": ori_img_base64,
        "output": output_base64,
        "class_counts":class_counts_json,
    }

    db = SessionLocal()
    image_record = db_models.ImageRecord(
        input_image_path=input_image_path,
        segmented_image_path=segmented_image_path,
        model=model_choice,

        Background_pct = class_counts[class_counts['class'] == 'Background']['percentage'].values[0],
        Building_pct = class_counts[class_counts['class'] == 'Building']['percentage'].values[0],
        Woodland_pct = class_counts[class_counts['class'] == 'Woodland']['percentage'].values[0],
        Water_pct = class_counts[class_counts['class'] == 'Water']['percentage'].values[0],
        Road_pct = class_counts[class_counts['class'] == 'Road']['percentage'].values[0],

        Background_count = class_counts[class_counts['class'] == 'Background']['pixel_count'].values[0],
        Building_count = class_counts[class_counts['class'] == 'Building']['pixel_count'].values[0],
        Woodland_count = class_counts[class_counts['class'] == 'Woodland']['pixel_count'].values[0],
        Water_count = class_counts[class_counts['class'] == 'Water']['pixel_count'].values[0],
        Road_count = class_counts[class_counts['class'] == 'Road']['pixel_count'].values[0],
    )
    db.add(image_record)
    db.commit()
    db.close()

    return JSONResponse(content=response_data)

@app.get("/get_data")
async def get_data(db: Session = Depends(get_db)):
    data = db.query(db_models.ImageRecord).all()
    result = [{"img_path": item.input_image_path, "output_path": item.segmented_image_path, "timestap": item.insertion_time} for item in data]
    return result

@app.get("/fetch_image_data")
async def fetch_image_data(db: Session = Depends(get_db)):
    data = db.query(db_models.ImageRecord).all()
    ret_json = []
    # result = [{"img_path": item.input_image_path, "output_path": item.segmented_image_path, "timestamp": item.insertion_time} for item in data]
    for item in data:
        byte_io_ori = io.BytesIO()
        img_ori = Image.open(item.input_image_path)
        img_ori.save(byte_io_ori, format="JPEG")
        ori_img_bytes = byte_io_ori.getvalue()
        ori_img_base64 = base64.b64encode(ori_img_bytes).decode()

        byte_io_output = io.BytesIO()
        img_out = Image.open(item.segmented_image_path)
        img_out.save(byte_io_output, format="JPEG")
        output_bytes = byte_io_output.getvalue()
        output_base64 = base64.b64encode(output_bytes).decode()

        ret_json.append({
            "img_id": item.id,
            "img_path":  ori_img_base64,
            "output_path": output_base64,
            "timestamp": [item.insertion_time.strftime("%d/%m/%Y"), item.insertion_time.strftime("%H:%M")],
            "model":item.model,
            "Background_pct":item.Background_pct,
            "Building_pct":item.Building_pct,
            "Woodland_pct":item.Woodland_pct,
            "Water_pct":item.Water_pct,
            "Road_pct":item.Road_pct,

            "Background_count":item.Background_count,
            "Building_count":item.Building_count,
            "Woodland_count":item.Woodland_count,
            "Water_count":item.Water_count,
            "Road_count":item.Road_count,
        })

        # print(ret_)
    
    return JSONResponse(content=ret_json)

@app.post("/fetch_mask_pred_data")
def fetch_mask_pred_data(prov_filter: str = Form(...)):

    with open('./map_metadata/metadata _with_province.json', 'r', encoding='utf8') as json_file:
        metadata_pos_prov = json.load(json_file)

    with open('./map_metadata/predicted_pixel_distributions_area.json', 'r', encoding='utf8') as json_file:
        metadata_area_prov = json.load(json_file)

    mask_names = list(metadata_pos_prov.keys())

    ret_json = []
    # result = [{"img_path": item.input_image_path, "output_path": item.segmented_image_path, "timestamp": item.insertion_time} for item in data]
    for ma_name in mask_names:

        if prov_filter != "all":
            if prov_filter != metadata_pos_prov[ma_name]["province"]:
                continue

        byte_io_ori = io.BytesIO()
        img_ori = Image.open(f"./predicted_masks/{ma_name}.png")
        img_ori.save(byte_io_ori, format="JPEG")
        ori_img_bytes = byte_io_ori.getvalue()
        ori_img_base64 = base64.b64encode(ori_img_bytes).decode()


        ret_json.append({
            "img_id": ma_name,
            "img_path":  ori_img_base64,
            "area": metadata_area_prov[ma_name]['total_pixel_area_meter'],

            "Background_pct":metadata_area_prov[ma_name]['pixel_pct_background'],
            "Building_pct":metadata_area_prov[ma_name]['pixel_pct_building'],
            "Woodland_pct":metadata_area_prov[ma_name]['pixel_pct_woodland'],
            "Water_pct":metadata_area_prov[ma_name]['pixel_pct_water'],
            "Road_pct":metadata_area_prov[ma_name]['pixel_pct_road'],

            "province":metadata_pos_prov[ma_name]['province']
        })

        # print(ret_)
    
    return JSONResponse(content=ret_json)

@app.get("/fetch_mask_pred_area_all_data")
def fetch_mask_pred_area_all_data():
    with open('./map_metadata/predicted_pixel_distributions_area.json', 'r', encoding='utf8') as json_file:
        metadata_area_prov = json.load(json_file)

    ret_json = {

        "area": metadata_area_prov["all"]['total_pixel_area_meter'],
        "Background_pct":metadata_area_prov["all"]['pixel_pct_background'],
        "Building_pct":metadata_area_prov["all"]['pixel_pct_building'],
        "Woodland_pct":metadata_area_prov["all"]['pixel_pct_woodland'],
        "Water_pct":metadata_area_prov["all"]['pixel_pct_water'],
        "Road_pct":metadata_area_prov["all"]['pixel_pct_road'],
    }

        # print(ret_)
    
    return JSONResponse(content=ret_json)
    

@app.delete("/delete_image/{img_id}")
def delete_image(img_id: int, db: Session = Depends(get_db)):
    image = db.query(db_models.ImageRecord).filter(db_models.ImageRecord.id == img_id).first()

    # if not image:
    #     raise HTTPException(status_code=404, detail="Image not found")

    try:
        # Delete the image record from the database
        db.delete(image)
        db.commit()
        # Delete the image files from the storage
        os.remove(image.input_image_path)
        os.remove(image.segmented_image_path)

        return {"status": "Image deleted successfully"}
    except Exception as e:
        # raise HTTPException(status_code=500, detail=str(e))
        pass



# Mount the static folder to serve HTML file
app.mount("/map", StaticFiles(directory="map"), name="map")


@app.post("/map")
def read_html(prov_filter: str = Form(...)):

    html_path = os.path.join("map", f"map_prov_{prov_filter}_down4.html")
    # html_path = os.path.join("map", f"test_all_down4.html")
    return FileResponse(html_path)