#!/bin/bash
for entry in ./*;
do
	cd $entry
	mv ./image_info/* ./
	rm -rf image_info
	cd ..
done
