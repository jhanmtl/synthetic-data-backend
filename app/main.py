from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import base64
import cv2
import numpy as np
import imgController as controller

# comment test

class RawForm(BaseModel):
    shape: str
    shapeColor: str
    text: str
    textColor: str
    sizeRange: List
    blurRange: List
    edgeDecay: List
    textOpacity: List
    xRange: List
    yRange: List
    angleRange: List
    overallOpacity: List
    numImg: int
    drawBbox: bool


def read_range(range_list):
    start = range_list[0]
    end = range_list[1]
    if start == end:
        end += 1
    return np.random.randint(start, end)


def read_str(text):
    if text == 'random' or text == "":
        return None
    else:
        return text


class ParsedForm:
    def __init__(self, form: RawForm):
        self.x = read_range(form.xRange)
        self.y = read_range(form.yRange)
        self.scolor = read_str(form.shapeColor)
        self.tcolor = read_str(form.textColor)
        self.text = read_str(form.text)
        self.shape = read_str(form.shape)
        self.angle = np.random.randint(form.angleRange[0], form.angleRange[1])

        if form.sizeRange[0] == form.sizeRange[1]:
            self.size_range = [form.sizeRange[0], form.sizeRange[0] + 1]
        else:
            self.size_range = form.sizeRange

        if form.edgeDecay[0] == form.edgeDecay[1]:
            self.edge_range = [form.edgeDecay[0], form.edgeDecay[0] + 1]
        else:
            self.edge_range = form.edgeDecay

        if form.blurRange[0] == form.blurRange[1]:
            self.shape_blur = [form.blurRange[0], form.blurRange[0] + 1]
        else:
            self.shape_blur = form.blurRange

        self.text_blur = [1, 3]
        self.text_opacity = form.textOpacity
        self.overall_opacity = form.overallOpacity
        self.num_imgs = form.numImg
        self.drawbbox = form.drawBbox


has_img = False
placeholder = np.ones((224, 224, 3)) * 225
placeholder = placeholder.astype(np.uint8)

app = FastAPI()
origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = controller.Generator(224, 224)


@app.get("/")
async def index():
    return {
        "year": 2020,
        "location": "montreal"
    }


@app.post("/post-form/")
async def read_form(rawform: RawForm):
    form = ParsedForm(rawform)
    generator.choose_color(form.scolor, form.tcolor)
    generator.choose_masks(form.shape, form.text)
    generator.scale(form.size_range)
    generator.rotate(form.angle)
    generator.preplace(form.x, form.y)
    generator.feather(form.edge_range, form.shape_blur, form.text_blur)
    generator.set_text_opacity(np.random.uniform(form.text_opacity[0], form.text_opacity[1]))
    generator.apply_mask(form.overall_opacity)

    img = generator.composite

    x1 = generator.info['x1'].item()
    y1 = generator.info['y1'].item()
    x2 = generator.info['x2'].item()
    y2 = generator.info['y2'].item()

    if form.drawbbox:
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

    _, jpg = cv2.imencode(".jpg", img)
    encode = (base64.b64encode(jpg)).decode()
    encode = "data:image/jpg;base64," + encode

    sendinfo = {"shape": generator.info['shape_label'],
                "shape color": generator.info['shape_color_label'],
                "text": generator.info['text_label'],
                "text color": generator.info['text_color_label'],
                "text heading": "{} {}".format(generator.info['text_heading'], chr(176)),
                "upper corner": "({},{})".format(x1, y1),
                "lower corner": "({},{})".format(x2, y2)
                }
    # print(json.dumps(sendinfo))
    return {"img": encode, "info": sendinfo}
