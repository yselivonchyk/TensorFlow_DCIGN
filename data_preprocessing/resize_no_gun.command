#!/bin/bash
#-type Grayscale
#cd $d && mogrify -path $stringA -type Grayscale  -resize 160x120 -crop 80x80+40+0  -quality 100  *.jpg
rm -R tmp
mkdir tmp
find . -mindepth 2 -maxdepth 2 -type d -print | grep -v 'tmp' | cpio -pd ./tmp

up=$"../../../tmp/"

find . -mindepth 3 -maxdepth 3 -type d -print | grep -v 'tmp' | while read d; do
  stringA=$up$d
  len=${#stringA}
  len=$((len-7))
  echo ${stringA}
  stringA=${stringA::len}
  (cd $d && mogrify -path $stringA -resize 160x120 -crop 160x80+0+0  -quality 100  *.jpg)
done

chmod 777 tmp
