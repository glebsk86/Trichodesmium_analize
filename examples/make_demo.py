"""Generate SYNTHETIC examples. These are not microscopy photographs."""
from pathlib import Path
import argparse

import cv2
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser()
    p.add_argument("destination",type=Path)
    args=p.parse_args()
    args.destination.mkdir(parents=True,exist_ok=False)
    rgb=np.full((600,800,3),242,np.uint8)
    cv2.line(rgb,(70,330),(730,330),(25,25,25),2)
    for x in range(76,730,12): cv2.line(rgb,(x,316),(x,344),(25,25,25),2)
    cv2.line(rgb,(420,60),(420,540),(25,25,25),2)
    for y in range(64,540,12): cv2.line(rgb,(406,y),(434,y),(25,25,25),2)
    rgb[100:112,90:270]=(165,145,65)
    for x in range(102,269,12): rgb[100:112,x]=(50,45,30)
    coords=np.array([[510,130],[540,145],[580,165],[620,180],[660,180]],np.int32)
    cv2.polylines(rgb,[coords],False,(170,145,70),12,cv2.LINE_AA)
    cv2.putText(rgb,"SYNTHETIC / NOT A MICROGRAPH",(30,585),cv2.FONT_HERSHEY_SIMPLEX,.6,(70,70,70),1)
    Image.fromarray(rgb).save(args.destination/"synthetic_ruler_and_rods.png")
    Image.fromarray(np.full((600,800,3),242,np.uint8)).save(args.destination/"synthetic_blank.png")


if __name__=="__main__": main()
