#!/bin/bash
mkdir tmp
find . -type d -depth 2 -print | grep -v 'tmp' | cpio -pd ./tmp

up=$"../../../tmp/"

find . -type d -depth 3 -print | grep -v 'tmp' | while read d; do
  stringA=$up$d
  len=${#stringA}
  len=$((len-7))
  echo ${stringA}
  stringA=${stringA::len}
  (cd $d && mogrify -path $stringA  -resize 48x32 -crop 32x32+0+0  -quality 100  *.jpg)
done
