#![allow(clippy::identity_op)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::eq_op)]
#![allow(dead_code)]
#![allow(unused_variables)] // TODO: Remove this in the future
pub mod emulator;
mod expr;
pub mod prover;
pub mod simulator;
#[cfg(test)]
mod tests;
mod utils;

use std::fmt::{self, Display};
use std::ops::{Index, IndexMut};

const REG_SP: usize = 2;
// First virtual register
const REG_V0: usize = 32;
const NUM_VIRT_REGS: usize = 4;

#[derive(Debug, Clone)]
pub struct Registers<T> {
    r: [T; 32 + NUM_VIRT_REGS],
    zero: T,
}

impl<T: Default> Default for Registers<T> {
    fn default() -> Self {
        Self {
            r: [0; 32 + NUM_VIRT_REGS].map(|_| T::default()),
            zero: T::default(),
        }
    }
}

impl Registers<u64> {
    pub fn into_f<F: From<u64>>(self) -> Registers<F> {
        Registers {
            r: self.r.map(|v| F::from(v)),
            zero: F::from(self.zero),
        }
    }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RW {
    Read = 0,
    Write = 1,
}

// Memory operation (read or write)
#[derive(Debug, Clone)]
pub struct MemOp {
    addr: u64,
    t: u64,
    rw: RW,
    value: u8,
}

#[derive(Default, Debug, Clone, Copy)]
pub enum VirtualOpcode {
    AssertLtu,
    AssertLte,
    AssertEq,
    #[default]
    Advice,
    Mul,
    Add,
}

#[derive(Default, Debug, Clone, Copy)]
pub enum Opcode {
    // RISC-U supported opcodes
    Lui,
    Addi,
    Ld,
    Sd,
    #[default]
    Add,
    Sub,
    Mul,
    Divu,
    Remu,
    Sltu,
    Beq,
    Jal,
    Jalr,
    Ecall,
    // Virtual
    Virtual(VirtualOpcode),
}

impl Opcode {
    fn from_usize(value: usize) -> Self {
        match value {
            0 => Self::Lui,
            1 => Self::Addi,
            2 => Self::Ld,
            3 => Self::Sd,
            4 => Self::Add,
            5 => Self::Sub,
            6 => Self::Mul,
            7 => Self::Divu,
            8 => Self::Remu,
            9 => Self::Sltu,
            10 => Self::Beq,
            11 => Self::Jal,
            12 => Self::Jalr,
            13 => Self::Ecall,
            14 => panic!("Virtual opcode"),
            _ => panic!("Undefined")
        }
    }
}

impl From<Opcode> for usize {
    fn from(value: Opcode) -> Self {
        match value {
            Opcode::Lui => 0,
            Opcode::Addi => 1,
            Opcode::Ld => 2,
            Opcode::Sd => 3,
            Opcode::Add => 4,
            Opcode::Sub => 5,
            Opcode::Mul => 6,
            Opcode::Divu => 7,
            Opcode::Remu => 8,
            Opcode::Sltu => 9,
            Opcode::Beq => 10,
            Opcode::Jal => 11,
            Opcode::Jalr => 12,
            Opcode::Ecall => 13,
            Opcode::Virtual(_) => panic!("Cannot convert"),
        }
    }
}

impl Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Default, Debug, Clone)]
pub struct Instruction {
    pub op: Opcode,
    rd: usize,
    rs1: usize,
    rs2: usize,
    imm: i64,
}

impl Instruction {
    fn new(op: Opcode, rd: usize, rs1: usize, rs2: usize, imm: i64) -> Self {
        Instruction {
            op,
            rd,
            rs1,
            rs2,
            imm,
        }
    }
}

#[derive(Debug)]
pub struct Step<T, I> {
    pub pc: T,
    pub vpc: T,
    pub inst: I,
    pub regs: Registers<T>,
    pub mem_t: u64,
    pub mem_ops: Vec<MemOp>,
    pub advice: u64,
}
