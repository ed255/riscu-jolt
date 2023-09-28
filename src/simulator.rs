//! Implementation of a zk circuit simulator of RISC-U (RV64I subset) following the Jolt approach:
//! implementing the verification of each instruction via Lasso decomposable tables with some small
//! arithmetic constraints.

use crate::Cpu;

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

impl LookupTables {
    // Take an input of 65 bits and return the lowest 64 bits.  This can be used to remove the
    // overflow bit after an addition.
    fn u65_to_u64(src: F) -> F {
        // println!("DBG {src} -> {:?}", src.0.to_bits_le());
        let src_bits = src.to_le_bits();
        let mut result = F::ZERO;
        for i in 0..64 {
            result = result + F::from(2).pow([i, 0, 0, 0]) * src_bits[i as usize];
            // println!("DBG {i} -> {}", result);
        }
        result
    }
}

pub type Simulator = Cpu<F>;

// Simulated zk circuit instructions with Lasso lookups
impl Cpu<F> {
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
        // Index. z = x + y over the native field
        let z = self.regs[rs1] + self.regs[rs2];
        println!("DBG z = {z}");
        // MLE. z has W+1 bits.  Take lowest W bits via lookup table
        let result = LookupTables::u65_to_u64(z);
        self.regs[rd] = result;
        self.pc = self.pc + F::from(4);
    }
    // `sub rd,rs1,rs2`: `rd = rs1 - rs2; pc = pc + 4`
    pub fn t_sub(&mut self, rd: usize, rs1: usize, rs2: usize) {
        // Index. z = x + (2^W - y) over the native field
        let z = self.regs[rs1] + (F::from(2).pow([64, 0, 0, 0]) - self.regs[rs2]);
        // MLE. z has W+1 bits.  Take lowest W bits via lookup table
        let result = LookupTables::u65_to_u64(z);
        self.regs[rd] = result;
        self.pc = self.pc + F::from(4);
    }
    // `mul rd,rs1,rs2`: `rd = rs1 * rs2; pc = pc + 4`
    pub fn t_mul(&mut self, rd: usize, rs1: usize, rs2: usize) {
        todo!();
    }
    // `divu rd,rs1,rs2`: `rd = rs1 / rs2; pc = pc + 4` where the values of `rs1` and `rs2` are interpreted as unsigned integers.
    pub fn t_divu(&mut self, rd: usize, rs1: usize, rs2: usize) {
        todo!();
    }
    // `remu rd,rs1,rs2`: `rd = rs1 % rs2; pc = pc + 4` where the values of `rs1` and `rs2` are interpreted as unsigned integers.
    pub fn t_remu(&mut self, rd: usize, rs1: usize, rs2: usize) {
        todo!();
    }
    // #### Comparison

    // `sltu rd,rs1,rs2`: `if (rs1 < rs2) { rd = 1 } else { rd = 0 } pc = pc + 4` where the values of `rs1` and `rs2` are interpreted as unsigned integers.
    pub fn t_sltu(&mut self, rd: usize, rs1: usize, rs2: usize) {
        todo!();
    }
    // #### Control

    // `beq rs1,rs2,imm`: `if (rs1 == rs2) { pc = pc + imm } else { pc = pc + 4 }` with `-2^12 <= imm < 2^12` and `imm % 2 == 0`
    // TODO
    // `jal rd,imm`: `rd = pc + 4; pc = pc + imm` with `-2^20 <= imm < 2^20` and `imm % 2 == 0`
    // TODO
    // `jalr rd,imm(rs1)`: `tmp = ((rs1 + imm) / 2) * 2; rd = pc + 4; pc = tmp` with `-2^11 <= imm < 2^11`
    // TODO
    // #### System

    // `ecall`: system call number is in `a7`, actual parameters are in `a0-a3`, return value is in `a0`.
    // TODO
}
