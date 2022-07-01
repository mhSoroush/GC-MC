#!/bin/bash

# Book 1M
python train.py -d book --data_seed 1234 --accum sum -do 0.7 -nsym -nb 2 -e 3500 --testing > ml_1m_testing.txt 2>&1



