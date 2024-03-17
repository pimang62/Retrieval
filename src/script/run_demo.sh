#!/bin/bash

nohup streamlit run ../demo/run_demo.py --server.address 0.0.0.0 --server.port 9100 --server.fileWatcherType none > demo.log 2>&1 &

# 50902