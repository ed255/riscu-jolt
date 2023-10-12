# Selfie

Build instructions:
```
git submodule init
git submodule update --init
cd ext/selfie
make selfie
```

## Build an example source code

```
./ext/selfie/selfie -c riscu_examples/c/fibo.c -o riscu_examples/c/fibo.bin -s riscu_examples/c/fibo.s 
```

## Run a risc-u binary under the selfie emulator

```
./ext/selfie/selfie -l riscu_examples/c/fibo.bin -m 1
```

The returned value from main can be seen in the output as `exit code X`.
