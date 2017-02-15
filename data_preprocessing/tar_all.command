ext=$".tar.gz"
find ./tmp/ -type d -maxdepth 1 -mindepth 1 | while read d; do
   tar -zcvf $d$ext $d/
done