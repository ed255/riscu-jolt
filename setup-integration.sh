#!/bin/sh
#
set +x
set +e

# Init selfie git submodule
git submodule init
git submodule update --init

# Build selfie
cd ext/selfie
make selfie
cd ../../

# Compile examples
./ext/selfie/selfie -c riscu_examples/c/fibo.c -o riscu_examples/c/fibo.bin -s riscu_examples/c/fibo.s
