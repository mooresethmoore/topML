import os
import sys


def saveGifNum(saveNameBase,num,res,fps,outName,tryRm=True, delFileCheck = lambda f: f.find(".png")!=-1 or f.find(".jpg")!=-1 or f.find(".json")!=-1):
    osout=os.system(f"ffmpeg -r {fps} -f image2 -s {res[0]}x{res[1]} -i {saveNameBase}t%d.png -vcodec libx264 -crf 18 {outName}.gif")
    if osout==0 and tryRm:
        rootDir=saveNameBase[:-1* saveNameBase[::-1].find("/")]
        for f in os.listdir(rootDir):
            if delFileCheck(f):
                try:
                    os.remove(f"{rootDir}{f}")
                except:
                    print(f"del error! \t {rootDir}{f}\n\n")
                    break