cat loss.csv |sed -e 's/\]//g' | sed -e 's/\[//g' > out
cp out loss.csv

