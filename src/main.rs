// https://github.com/cksystemsteaching/selfie/blob/main/riscu.md

// RISC-U is a subset of RV64I

use std::ops::{Index, IndexMut};

// Emulator

struct Registers<T> {
    r: [T; 32],
    zero: T,
}

impl<T> Index<usize> for Registers<T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        &self.r[i]
    }
}

impl<T> IndexMut<usize> for Registers<T> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        if i == 0 {
            &mut self.zero
        } else {
            &mut self.r[i]
        }
    }
}

struct Cpu<T> {
    pc: T,
    regs: Registers<T>,
}

// Emulated instructions
impl Cpu<u64> {
    // #### Initialization

    // `lui rd,imm`: `rd = imm * 2^12; pc = pc + 4` with `-2^19 <= imm < 2^19`
    //
    // `addi rd,rs1,imm`: `rd = rs1 + imm; pc = pc + 4` with `-2^11 <= imm < 2^11`
    //
    // #### Memory

    // `ld rd,imm(rs1)`: `rd = memory[rs1 + imm]; pc = pc + 4` with `-2^11 <= imm < 2^11`
    //
    // `sd rs2,imm(rs1)`: `memory[rs1 + imm] = rs2; pc = pc + 4` with `-2^11 <= imm < 2^11`
    //
    // #### Arithmetic

    // `add rd,rs1,rs2`: `rd = rs1 + rs2; pc = pc + 4`
    fn add(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let (result, _) = self.regs[rs1].overflowing_add(self.regs[rs2]);
        self.regs[rd] = result;
        self.pc = self.pc + 4;
    }
    // `sub rd,rs1,rs2`: `rd = rs1 - rs2; pc = pc + 4`
    fn sub(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let (result, _) = self.regs[rs1].overflowing_sub(self.regs[rs2]);
        self.regs[rd] = result;
        self.pc = self.pc + 4;
    }
    // `mul rd,rs1,rs2`: `rd = rs1 * rs2; pc = pc + 4`
    fn mul(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let (result, _) = self.regs[rs1].overflowing_mul(self.regs[rs2]);
        self.regs[rd] = result;
        self.pc = self.pc + 4;
    }
    // `divu rd,rs1,rs2`: `rd = rs1 / rs2; pc = pc + 4` where the values of `rs1` and `rs2` are interpreted as unsigned integers.
    fn divu(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let (result, _) = self.regs[rs1].overflowing_div(self.regs[rs2]);
        self.regs[rd] = result;
        self.pc = self.pc + 4;
    }
    // `remu rd,rs1,rs2`: `rd = rs1 % rs2; pc = pc + 4` where the values of `rs1` and `rs2` are interpreted as unsigned integers.
    fn remu(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let (result, _) = self.regs[rs1].overflowing_rem(self.regs[rs2]);
        self.regs[rd] = result;
        self.pc = self.pc + 4;
    }
    // #### Comparison

    // `sltu rd,rs1,rs2`: `if (rs1 < rs2) { rd = 1 } else { rd = 0 } pc = pc + 4` where the values of `rs1` and `rs2` are interpreted as unsigned integers.
    fn sltu(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let result = if self.regs[rs1] < self.regs[rs2] {
            1
        } else {
            0
        };
        self.regs[rd] = result;
        self.pc = self.pc + 4;
    }
    // #### Control
    //
    // `beq rs1,rs2,imm`: `if (rs1 == rs2) { pc = pc + imm } else { pc = pc + 4 }` with `-2^12 <= imm < 2^12` and `imm % 2 == 0`
    //
    // `jal rd,imm`: `rd = pc + 4; pc = pc + imm` with `-2^20 <= imm < 2^20` and `imm % 2 == 0`
    //
    // `jalr rd,imm(rs1)`: `tmp = ((rs1 + imm) / 2) * 2; rd = pc + 4; pc = tmp` with `-2^11 <= imm < 2^11`
    //
    // #### System

    // `ecall`: system call number is in `a7`, actual parameters are in `a0-a3`, return value is in `a0`.
}

use ark_bn254::fr::Fr as F;
use ark_ff::{biginteger::BigInteger, Field};

trait ToLeBits {
    type T;

    fn to_le_bits(&self) -> Vec<Self::T>;
}

impl ToLeBits for u64 {
    type T = u64;
    fn to_le_bits(&self) -> Vec<Self::T> {
        let mut result = vec![0u64; 64];
        for i in 0..64 {
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
        self.0
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
        let src_bits = src.to_le_bits();
        let mut result = F::ZERO;
        for i in 0..64 {
            result = result + F::from(2).pow([i, 0, 0, 0]) * src_bits[i as usize]
        }
        result
    }
}

// Emulated instructions with Lasso lookups
impl Cpu<F> {
    // `add rd,rs1,rs2`: `rd = rs1 + rs2; pc = pc + 4`
    fn t_add(&mut self, rd: usize, rs1: usize, rs2: usize) {
        // Index. z = x + y over the native field
        let z = self.regs[rs1] + self.regs[rs2];
        // MLE. z has W+1 bits.  Take lowest W bits via lookup table
        let result = LookupTables::u65_to_u64(z);
        self.regs[rd] = result;
        self.pc = self.pc + F::from(4);
    }
    // `sub rd,rs1,rs2`: `rd = rs1 - rs2; pc = pc + 4`
    fn t_sub(&mut self, rd: usize, rs1: usize, rs2: usize) {
        // Index. z = x + (2^W - y) over the native field
        let z = self.regs[rs1] + (F::from(2).pow([64, 0, 0, 0]) - self.regs[rs2]);
        // MLE. z has W+1 bits.  Take lowest W bits via lookup table
        let result = LookupTables::u65_to_u64(z);
        self.regs[rd] = result;
        self.pc = self.pc + F::from(4);
    }
    // `mul rd,rs1,rs2`: `rd = rs1 * rs2; pc = pc + 4`
    fn t_mul(&mut self, rd: usize, rs1: usize, rs2: usize) {
        unimplemented!();
    }
}

fn main() {
    println!("Hello, world!");
}
