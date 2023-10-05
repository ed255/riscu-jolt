// Fibonacci sequencey starting with [1, 1]:
// [1, 1, 2, 3, 5, 8, 13, ... ]
uint64_t fib(uint64_t n) {
	uint64_t i;
	uint64_t tmp;
	uint64_t a;
	uint64_t b;
	a = 0;
	b = 1;
	while (i < n) {
		tmp = b;
		b = a + b;
		a = tmp;
		i = i + 1;
	}
	return b;
}

uint64_t main() {
	uint64_t n;
	n = 10;
	return fib(n);
}
