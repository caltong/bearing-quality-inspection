#!/bin/bash
for i in '0.2' '0.4' '0.6' '0.8' '1.0'
do
    python generate_side_model.py $i 0.8
done
