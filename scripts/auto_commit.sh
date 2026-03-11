#!/bin/bash

while true
do
    git add .
    git commit -m "auto data update $(date)"
    git push
    sleep 300
done
