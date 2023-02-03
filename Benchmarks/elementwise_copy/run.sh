#!/bin/bash

compile.sh elementwise-copy
./elementwise-copy.out 1024 >> results.txt
./elementwise-copy.out 2048 >> results.txt
./elementwise-copy.out 4096 >> results.txt
./elementwise-copy.out 8192 >> results.txt
./elementwise-copy.out 16384 >> results.txt
./elementwise-copy.out 32768 >> results.txt
./elementwise-copy.out 65536 >> results.txt
./elementwise-copy.out 131072 >> results.txt
./elementwise-copy.out 262144 >> results.txt
./elementwise-copy.out 524288 >> results.txt
./elementwise-copy.out 1048576 >> results.txt
./elementwise-copy.out 2097152 >> results.txt
./elementwise-copy.out 4194304 >> results.txt
./elementwise-copy.out 8388608 >> results.txt
./elementwise-copy.out 16777216 >> results.txt