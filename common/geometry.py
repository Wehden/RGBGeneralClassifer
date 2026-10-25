"""Identical crop bounds and image fingerprints on the board and service."""
import math


def bounds(box, width, height, padding=0):
    y1,x1,y2,x2=box
    dy,dx=(y2-y1)*padding,(x2-x1)*padding
    return (max(0,math.floor((x1-dx)*width)),max(0,math.floor((y1-dy)*height)),min(width,math.ceil((x2+dx)*width)),min(height,math.ceil((y2+dy)*height)))


def fingerprint(pixel, width, height):
    value=0
    for row in range(8):
        samples=[]
        y=min(height-1,int((row+.5)*height/8))
        for col in range(9):
            x=min(width-1,int((col+.5)*width/9))
            r,g,b=map(int,pixel(x,y))
            samples.append(299*r+587*g+114*b)
        for a,b in zip(samples,samples[1:]):
            value=(value<<1)|int(a>b)
    return f'{value:016x}'


def iou(a,b):
    overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap
    return overlap/union if union else 0
