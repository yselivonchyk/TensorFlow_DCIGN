mkdir 32_32
cd 640_480
mogrify -path ./../32_32  -resize 48x32 -crop 32x32+0+0  -quality 100  *.jpg