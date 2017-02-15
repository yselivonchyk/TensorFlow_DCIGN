#!/bin/bash
#-type Grayscale
#cd $d && mogrify -path $stringA -type Grayscale  -resize 160x120 -crop 80x80+40+0  -quality 100  *.jpg
rm -R tmp_grey
mkdir tmp_grey


find . -mindepth 2 -maxdepth 2 -type d -print | grep -v 'tmp' | cpio -pd ./tmp_grey/

up=$"../../../tmp_grey/"

find . -mindepth 3 -maxdepth 3 -type d -print | grep -v 'tmp' | while read d; do
  stringA=$up$d
  len=${#stringA}
  len=$((len-7))
  echo ${stringA}
  stringA=${stringA::len}
  # echo ${stringA}
  (cd $d && mogrify -path $stringA -resize 160x120 -crop 40x40+60+40 -fx '(r+g+b)/3' -quality 100  *.jpg)
done

chmod 777 tmp
