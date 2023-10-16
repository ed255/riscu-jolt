// sha256 is difficult to implement in C-star because this language only supports 64 bit arithmetic, whereas sha256 uses 32 bit arithmetic everywhere.  On 32 bit arithmetic, the only challenging part is doing `rightrotate`, `and` and `xor`.
uint64_t h;uint64_t h1; uint64_t h2; uint64_t h3;
uint64_t h4; uint64_t h5; uint64_t h6; uint64_t h7;

uint64_t k; uint64_t k1; uint64_t k2; uint64_t k3;
uint64_t k4; uint64_t k5; uint64_t k6; uint64_t k7;
uint64_t k8; uint64_t k9; uint64_t k10; uint64_t k11;
uint64_t k12; uint64_t k13; uint64_t k14; uint64_t k15;
uint64_t k16; uint64_t k17; uint64_t k18; uint64_t k19;
uint64_t k20; uint64_t k21; uint64_t k22; uint64_t k23;
uint64_t k24; uint64_t k25; uint64_t k26; uint64_t k27;
uint64_t k28; uint64_t k29; uint64_t k30; uint64_t k31;
uint64_t k32; uint64_t k33; uint64_t k34; uint64_t k35;
uint64_t k36; uint64_t k37; uint64_t k38; uint64_t k39;
uint64_t k40; uint64_t k41; uint64_t k42; uint64_t k43;
uint64_t k44; uint64_t k45; uint64_t k46; uint64_t k47;
uint64_t k48; uint64_t k49; uint64_t k50; uint64_t k51;
uint64_t k52; uint64_t k53; uint64_t k54; uint64_t k55;
uint64_t k56; uint64_t k57; uint64_t k58; uint64_t k59;
uint64_t k60; uint64_t k61; uint64_t k62; uint64_t k63;

uint64_t w; uint64_t w1; uint64_t w2; uint64_t w3;
uint64_t w4; uint64_t w5; uint64_t w6; uint64_t w7;
uint64_t w8; uint64_t w9; uint64_t w10; uint64_t w11;
uint64_t w12; uint64_t w13; uint64_t w14; uint64_t w15;
uint64_t w16; uint64_t w17; uint64_t w18; uint64_t w19;
uint64_t w20; uint64_t w21; uint64_t w22; uint64_t w23;
uint64_t w24; uint64_t w25; uint64_t w26; uint64_t w27;
uint64_t w28; uint64_t w29; uint64_t w30; uint64_t w31;
uint64_t w32; uint64_t w33; uint64_t w34; uint64_t w35;
uint64_t w36; uint64_t w37; uint64_t w38; uint64_t w39;
uint64_t w40; uint64_t w41; uint64_t w42; uint64_t w43;
uint64_t w44; uint64_t w45; uint64_t w46; uint64_t w47;
uint64_t w48; uint64_t w49; uint64_t w50; uint64_t w51;
uint64_t w52; uint64_t w53; uint64_t w54; uint64_t w55;
uint64_t w56; uint64_t w57; uint64_t w58; uint64_t w59;
uint64_t w60; uint64_t w61; uint64_t w62; uint64_t w63;

// Initialize global constants
void init() {
	h = 1779033703; // 0x6a09e667
	h1 = 3144134277; // 0xbb67ae85
	h2 = 1013904242; // 0x3c6ef372
	h3 = 2773480762; // 0xa54ff53a
	h4 = 1359893119; // 0x510e527f
	h5 = 2600822924; // 0x9b05688c
	h6 = 528734635; // 0x1f83d9ab
	h7 = 1541459225; // 0x5be0cd19

	k = 1116352408; // 0x428a2f98
	k1 = 1899447441; // 0x71374491
	k2 = 3049323471; // 0xb5c0fbcf
	k3 = 3921009573; // 0xe9b5dba5
	k4 = 961987163; // 0x3956c25b
	k5 = 1508970993; // 0x59f111f1
	k6 = 2453635748; // 0x923f82a4
	k7 = 2870763221; // 0xab1c5ed5
	k8 = 3624381080; // 0xd807aa98
	k9 = 310598401; // 0x12835b01
	k10 = 607225278; // 0x243185be
	k11 = 1426881987; // 0x550c7dc3
	k12 = 1925078388; // 0x72be5d74
	k13 = 2162078206; // 0x80deb1fe
	k14 = 2614888103; // 0x9bdc06a7
	k15 = 3248222580; // 0xc19bf174
	k16 = 3835390401; // 0xe49b69c1
	k17 = 4022224774; // 0xefbe4786
	k18 = 264347078; // 0xfc19dc6
	k19 = 604807628; // 0x240ca1cc
	k20 = 770255983; // 0x2de92c6f
	k21 = 1249150122; // 0x4a7484aa
	k22 = 1555081692; // 0x5cb0a9dc
	k23 = 1996064986; // 0x76f988da
	k24 = 2554220882; // 0x983e5152
	k25 = 2821834349; // 0xa831c66d
	k26 = 2952996808; // 0xb00327c8
	k27 = 3210313671; // 0xbf597fc7
	k28 = 3336571891; // 0xc6e00bf3
	k29 = 3584528711; // 0xd5a79147
	k30 = 113926993; // 0x6ca6351
	k31 = 338241895; // 0x14292967
	k32 = 666307205; // 0x27b70a85
	k33 = 773529912; // 0x2e1b2138
	k34 = 1294757372; // 0x4d2c6dfc
	k35 = 1396182291; // 0x53380d13
	k36 = 1695183700; // 0x650a7354
	k37 = 1986661051; // 0x766a0abb
	k38 = 2177026350; // 0x81c2c92e
	k39 = 2456956037; // 0x92722c85
	k40 = 2730485921; // 0xa2bfe8a1
	k41 = 2820302411; // 0xa81a664b
	k42 = 3259730800; // 0xc24b8b70
	k43 = 3345764771; // 0xc76c51a3
	k44 = 3516065817; // 0xd192e819
	k45 = 3600352804; // 0xd6990624
	k46 = 4094571909; // 0xf40e3585
	k47 = 275423344; // 0x106aa070
	k48 = 430227734; // 0x19a4c116
	k49 = 506948616; // 0x1e376c08
	k50 = 659060556; // 0x2748774c
	k51 = 883997877; // 0x34b0bcb5
	k52 = 958139571; // 0x391c0cb3
	k53 = 1322822218; // 0x4ed8aa4a
	k54 = 1537002063; // 0x5b9cca4f
	k55 = 1747873779; // 0x682e6ff3
	k56 = 1955562222; // 0x748f82ee
	k57 = 2024104815; // 0x78a5636f
	k58 = 2227730452; // 0x84c87814
	k59 = 2361852424; // 0x8cc70208
	k60 = 2428436474; // 0x90befffa
	k61 = 2756734187; // 0xa4506ceb
	k62 = 3204031479; // 0xbef9a3f7
	k63 = 3329325298; // 0xc67178f2
}

void do_chunk(uint64_t* input, uint64_t input_len) {
	
}

uint64_t main() {
	uint64_t* input;
	uint64_t input_len;
	input = "hello";
	input_len = 5;
	do_chunk(input, input_len);
	return 0;
}
