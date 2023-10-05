//! Implementation of an emulator of RISC-U (RV64I subset)

use crate::Registers;

#[derive(Default)]
pub struct Emulator<T> {
    pub(crate) pc: T,
    pub(crate) regs: Registers<T>,
    pub(crate) mem: Vec<u8>,
}

// Emulated instructions
impl Emulator<u64> {
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
    pub fn add(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let (result, _) = self.regs[rs1].overflowing_add(self.regs[rs2]);
        self.regs[rd] = result;
        self.pc = self.pc + 4;
    }
    // `sub rd,rs1,rs2`: `rd = rs1 - rs2; pc = pc + 4`
    pub fn sub(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let (result, _) = self.regs[rs1].overflowing_sub(self.regs[rs2]);
        self.regs[rd] = result;
        self.pc = self.pc + 4;
    }
    // `mul rd,rs1,rs2`: `rd = rs1 * rs2; pc = pc + 4`
    pub fn mul(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let (result, _) = self.regs[rs1].overflowing_mul(self.regs[rs2]);
        self.regs[rd] = result;
        self.pc = self.pc + 4;
    }
    // `divu rd,rs1,rs2`: `rd = rs1 / rs2; pc = pc + 4` where the values of `rs1` and `rs2` are interpreted as unsigned integers.
    pub fn divu(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let (result, _) = if self.regs[rs2] == 0 {
            // From riscv-spec-20191213 section 7.2:
            // > The quotient of division by zero has all bits set
            (0xffffffffffffffff, false)
        } else {
            self.regs[rs1].overflowing_div(self.regs[rs2])
        };
        self.regs[rd] = result;
        self.pc = self.pc + 4;
    }
    // `remu rd,rs1,rs2`: `rd = rs1 % rs2; pc = pc + 4` where the values of `rs1` and `rs2` are interpreted as unsigned integers.
    pub fn remu(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let (result, _) = if self.regs[rs2] == 0 {
            // From riscv-spec-20191213 section 7.2:
            // > the remainder of division by zero equals the dividend
            (self.regs[rs1], false)
        } else {
            self.regs[rs1].overflowing_rem(self.regs[rs2])
        };
        self.regs[rd] = result;
        self.pc = self.pc + 4;
    }
    // #### Comparison

    // `sltu rd,rs1,rs2`: `if (rs1 < rs2) { rd = 1 } else { rd = 0 } pc = pc + 4` where the values of `rs1` and `rs2` are interpreted as unsigned integers.
    pub fn sltu(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let result = if self.regs[rs1] < self.regs[rs2] {
            1
        } else {
            0
        };
        self.regs[rd] = result;
        self.pc = self.pc + 4;
    }
    // #### Control

    // `beq rs1,rs2,imm`: `if (rs1 == rs2) { pc = pc + imm } else { pc = pc + 4 }` with `-2^12 <= imm < 2^12` and `imm % 2 == 0`
    pub fn beq(&mut self, rs1: usize, rs2: usize, imm: i64) {
        if self.regs[rs1] == self.regs[rs2] {
            assert!(imm < 2 << 12 && imm >= -(2 << 12));
            assert_eq!(imm % 2, 0);
            assert_eq!(imm % 4, 0, "instruction-address-misaligned");
            let pc = self.pc as i64;
            self.pc = (pc + imm as i64) as u64;
        } else {
            self.pc = self.pc + 4;
        }
    }
    // `jal rd,imm`: `rd = pc + 4; pc = pc + imm` with `-2^20 <= imm < 2^20` and `imm % 2 == 0`
    pub fn jal(&mut self, rd: usize, imm: i64) {
        assert!(imm < 2 << 20 && imm >= -(2 << 20));
        assert_eq!(imm % 2, 0);
        assert_eq!(imm % 4, 0, "instruction-address-misaligned");
        let pc = self.pc as i128;
        self.pc = (pc + imm as i128) as u64;
        self.regs[rd] = self.pc;
    }
    // `jalr rd,imm(rs1)`: `tmp = ((rs1 + imm) / 2) * 2; rd = pc + 4; pc = tmp` with `-2^11 <= imm < 2^11`
    pub fn jalr(&mut self, rd: usize, rs1: usize, imm: i64) {
        assert!(imm < 2 << 12 && imm >= -(2 << 12));
        let tmp = ((self.regs[rs1] as i128 + imm as i128) / 2) * 2;
        assert_eq!(tmp % 4, 0, "instruction-address-misaligned");
        self.regs[rd] = self.pc + 4;
        self.pc = tmp as u64;
    }

    // #### System

    // `ecall`: system call number is in `a7`, actual parameters are in `a0-a3`, return value is in `a0`.
    // TODO
}

pub enum Opcode {
    // RISC-U supported opcodes
    Lui,
    Addi,
    Ld,
    Sd,
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
}

pub struct Instruction {
    op: Opcode,
    rd: usize,
    rs1: usize,
    rs2: usize,
    imm: i64,
}

#[repr(u32)]
enum RvOpcode {
    LUI = 0b0110111,
    AUIPC = 0b0010111,
    JAL = 0b1101111,
    JALR = 0b1100111,
    // Conditional Branches
    BRANCH = 0b1100011, // BEQ, BNE, BLT, BGE, BLTU, BGEU
    LOAD = 0b0000011,   // LB, LH, LW, LBU, LHU
    STORE = 0b0100011,  // SB, SH, SW
    // Integer Register-Immediate Instructions
    OPIMM = 0b0010011, // ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI
    // Integer Register Register Operations
    OP = 0b0110011, // ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND
    MISC_MEM = 0b0001111,
    SYSTEM = 0b1110011, // ECALL, EBREAK, CSR*
}

pub fn decode(v: u32) -> Instruction {
    unimplemented!();
}
