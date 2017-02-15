#!/bin/bash 
echo hello world
mogrify -path ../32_32 -resize -thumbnail 32x32  *.png
mogrify   -path thumbnail-directory   -thumbnail 100x100  *

mkdir 32_32
cd 640_480
mogrify -path ./../32_32  -resize 48x32 -crop 32x32+0+0  -quality 100  *.jpg