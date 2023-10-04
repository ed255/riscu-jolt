//! Implementation of a zk circuit simulator of RISC-U (RV64I subset) following the Jolt approach:
//! implementing the verification of each instruction via Lasso decomposable tables with some small
//! arithmetic constraints.

use crate::Registers;

use ark_bn254::fr::Fr as F;
use ark_ff::{biginteger::BigInteger, Field, PrimeField};

trait ToLeBits {
    type T;

    fn to_le_bits(&self) -> Vec<Self::T>;
}

impl ToLeBits for u64 {
    type T = u64;
    fn to_le_bits(&self) -> Vec<Self::T> {
        let n = 64;
        let mut result = vec![0u64; n];
        for i in 0..n {
            if self & 1 << i != 0 {
                result[i] = 1;
            }
        }
        result
    }
}

impl ToLeBits for F {
    type T = F;
    fn to_le_bits(&self) -> Vec<Self::T> {
        self.into_bigint()
            .to_bits_le()
            .iter()
            .map(|bit| F::from(*bit))
            .collect()
    }
}

struct LookupTables {}

fn f_pow(base: usize, exp: usize) -> F {
    F::from(base as u64).pow([exp as u64, 0, 0, 0])
}

// Notation:
// - c: number of chunks
// - m: subtable size
// - W: word size (32 for RV32, 64 for RV64)
impl LookupTables {
    // Take an input of W+1 bits and return the lowest W bits.  This can be used to remove the
    // overflow bit after an addition.
    // Full Table
    fn wp1_to_w(w: usize, src: F) -> F {
        // src in {0, 1}^{W+1}
        assert!(src.into_bigint().num_bits() as usize <= w + 1);
        let src_bits = src.to_le_bits();
        let mut result = F::ZERO;
        for i in 0..w {
            result = result + f_pow(2, i) * src_bits[i as usize];
        }
        result
    }
    // Take an input of 2*W bits and return the lowest W bits.  This can be used to remove the
    // overflow bits after a multiplication.
    // Full Table
    fn wx2_to_w(w: usize, src: F) -> F {
        // src in {0, 1}^{2*W}
        assert!(src.into_bigint().num_bits() as usize <= 2 * w);
        let src_bits = src.to_le_bits();
        let mut result = F::ZERO;
        for i in 0..w {
            result = result + f_pow(2, i) * src_bits[i as usize];
        }
        result
    }
    // Equality comparison between two n bits inputs.  Outputs 1 if x == y, 0 otherwise.
    // FullTable EQ
    // Ref Jolt 4.2.1
    fn eq(w: usize, c: usize, x_y: F) -> F {
        // x||y in {0, 1}^{2*W}
        assert!(x_y.into_bigint().num_bits() as usize <= 2 * w);
        let x_y_bits = x_y.to_le_bits();
        let x = &x_y_bits[0..w];
        let y = &x_y_bits[w..2 * w];
        let mut result = F::ZERO;
        for i in 0..w / c {
            let x_i = &x[i * w / c..(i + 1) * w / c];
            let y_i = &y[i * w / c..(i + 1) * w / c];
            result *= Self::eq_mle(x_i, y_i);
        }
        result
    }

    // Evaluate equalty between two n bits inputs.  Outputs 1 if x == y, 0 otherwise.
    // SubTable EQ
    // Ref Jolt 4.2.1 (4)
    fn eq_mle(x: &[F], y: &[F]) -> F {
        assert_eq!(x.len(), y.len());
        let mut result = F::ONE;
        for i in 0..x.len() {
            result *= x[i] * y[i] + (F::ONE - x[i]) * (F::ONE - y[i]);
        }
        result
    }

    // Lower than comparison between two n bits inputs.  Outputs 1 if x < y, 0 otherwise.
    // FullTable LTU
    // Ref Jolt 4.2.2
    fn ltu(w: usize, c: usize, x_y: F) -> F {
        // x||y in {0, 1}^{2*W}
        assert!(x_y.into_bigint().num_bits() as usize <= 2 * w);
        let x_y_bits = x_y.to_le_bits();
        let x = &x_y_bits[0..w];
        let y = &x_y_bits[w..2 * w];
        let mut result = F::ZERO;
        let mut eq_acc = F::ONE;
        for i in (0..c).rev() {
            let x_i = &x[i * (w / c)..(i + 1) * (w / c)];
            let y_i = &y[i * (w / c)..(i + 1) * (w / c)];
            result += Self::ltu_mle(x_i, y_i) * eq_acc;
            eq_acc *= Self::eq_mle(x_i, y_i);
        }
        result
    }

    // Ref Jolt 4.2.2 (5)
    fn ltu_i_mle(i: usize, x: &[F], y: &[F]) -> F {
        assert_eq!(x.len(), y.len());
        (F::ONE - x[i]) * y[i] * Self::eq_mle(&x[i + 1..], &y[i + 1..])
    }

    // Evaluate lower than between two n bits inputs.  Outputs 1 if x < y, 0 otherwise.
    // SubTable LTU
    // Ref Jolt 4.2.2 (6)
    fn ltu_mle(x: &[F], y: &[F]) -> F {
        assert_eq!(x.len(), y.len());
        let mut result = F::ZERO;
        for i in 0..x.len() {
            result += Self::ltu_i_mle(i, x, y);
        }
        result
    }
}

#[derive(Default)]
pub struct Simulator {
    pub(crate) pc: F,
    pub(crate) regs: Registers<F>,
    // TODO: Memory simulation
    // TODO: W
    // TODO: C
}

// TODO, move these two constants to parameters of the object.
const W: usize = 64;
const C: usize = 4;

// Simulated zk circuit instructions with Lasso lookups
impl Simulator {
    // #### Initialization

    // `lui rd,imm`: `rd = imm * 2^12; pc = pc + 4` with `-2^19 <= imm < 2^19`
    // TODO
    // `addi rd,rs1,imm`: `rd = rs1 + imm; pc = pc + 4` with `-2^11 <= imm < 2^11`
    // TODO

    // #### Memory

    // `ld rd,imm(rs1)`: `rd = memory[rs1 + imm]; pc = pc + 4` with `-2^11 <= imm < 2^11`
    // TODO
    // `sd rs2,imm(rs1)`: `memory[rs1 + imm] = rs2; pc = pc + 4` with `-2^11 <= imm < 2^11`
    // TODO

    // #### Arithmetic

    // `add rd,rs1,rs2`: `rd = rs1 + rs2; pc = pc + 4`
    pub fn t_add(&mut self, rd: usize, rs1: usize, rs2: usize) {
        // Ref: Jolt 5.2
        // Index. z = x + y over the native field
        let z = self.regs[rs1] + self.regs[rs2];
        // MLE. z has W+1 bits.  Take lowest W bits via lookup table
        let result = LookupTables::wp1_to_w(W, z);
        self.regs[rd] = result;
        self.pc = self.pc + F::from(4);
    }
    // `sub rd,rs1,rs2`: `rd = rs1 - rs2; pc = pc + 4`
    pub fn t_sub(&mut self, rd: usize, rs1: usize, rs2: usize) {
        // Ref: Jolt 5.2
        // Index. z = x + (2^W - y) over the native field
        let z = self.regs[rs1] + (f_pow(2, W) - self.regs[rs2]);
        // MLE. z has W+1 bits.  Take lowest W bits via lookup table
        let result = LookupTables::wp1_to_w(W, z);
        self.regs[rd] = result;
        self.pc = self.pc + F::from(4);
    }
    // `mul rd,rs1,rs2`: `rd = rs1 * rs2; pc = pc + 4`
    pub fn t_mul(&mut self, rd: usize, rs1: usize, rs2: usize) {
        // Ref: Jolt 6.2.1
        // Index. z = x * y over the native field
        let z = self.regs[rs1] * self.regs[rs2];
        // MLE. z has 2*W bits.  Take lowest W bits via lookup table
        let result = LookupTables::wx2_to_w(W, z);
        self.regs[rd] = result;
        self.pc = self.pc + F::from(4);
    }
    // `divu rd,rs1,rs2`: `rd = rs1 / rs2; pc = pc + 4` where the values of `rs1` and `rs2` are
    // interpreted as unsigned integers.
    pub fn t_divu(&mut self, rd: usize, rs1: usize, rs2: usize) {
        // Ref: Jolt 6.3
        // x = q * y + r
        // where x = rs1, y = rs2, q = rd
        todo!();
    }
    // `remu rd,rs1,rs2`: `rd = rs1 % rs2; pc = pc + 4` where the values of `rs1` and `rs2` are
    // interpreted as unsigned integers.
    pub fn t_remu(&mut self, rd: usize, rs1: usize, rs2: usize) {
        // Ref: Jolt 6.3
        // x = q * y + r
        // where x = rs1, y = rs2, r = rd
        todo!();
    }

    // #### Comparison

    // `sltu rd,rs1,rs2`: `if (rs1 < rs2) { rd = 1 } else { rd = 0 } pc = pc + 4` where the values
    // of `rs1` and `rs2` are interpreted as unsigned integers.
    pub fn t_sltu(&mut self, rd: usize, rs1: usize, rs2: usize) {
        // Ref: Jolt 5.3, 4.2.2
        let result = LookupTables::ltu(W, C, self.regs[rs1] + f_pow(2, W) * self.regs[rs2]);
        self.regs[rd] = result;
        self.pc = self.pc + F::from(4);
    }
    // #### Control

    // `beq rs1,rs2,imm`: `if (rs1 == rs2) { pc = pc + imm } else { pc = pc + 4 }` with `-2^12 <=
    // imm < 2^12` and `imm % 2 == 0`
    // TODO
    // `jal rd,imm`: `rd = pc + 4; pc = pc + imm` with `-2^20 <= imm < 2^20` and `imm % 2 == 0`
    // TODO
    // `jalr rd,imm(rs1)`: `tmp = ((rs1 + imm) / 2) * 2; rd = pc + 4; pc = tmp` with `-2^11 <= imm
    // < 2^11`
    // TODO
    // #### System

    // `ecall`: system call number is in `a7`, actual parameters are in `a0-a3`, return value is in
    // `a0`.
    // TODO
}
