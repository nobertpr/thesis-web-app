from sqlalchemy import Column, Integer, String, DateTime, Float
from database import Base
from datetime import datetime

class ImageRecord(Base):
    __tablename__ = "image_records"

    id = Column(Integer, primary_key=True, index=True)
    input_image_path = Column(String)
    segmented_image_path = Column(String)
    insertion_time = Column(DateTime, default=datetime.now)
    model = Column(String)
    Background_pct = Column(Float)
    Building_pct = Column(Float)
    Woodland_pct = Column(Float)
    Water_pct = Column(Float)
    Road_pct = Column(Float)

    Background_count = Column(Float)
    Building_count = Column(Float)
    Woodland_count = Column(Float)
    Water_count = Column(Float)
    Road_count = Column(Float)