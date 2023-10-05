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

#[derive(Debug, Default)]
pub enum Opcode {
    // RISC-U supported opcodes
    #[default]
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

#[derive(Debug, Default)]
pub struct Instruction {
    op: Opcode,
    rd: usize,
    rs1: usize,
    rs2: usize,
    imm: i64,
}

const MASK_OPCODE: u32 = 0b00000000000000000000000001111111;
const MASK_FUNCT3: u32 = 0b00000000000000000111000000000000;
const FUNCT3_JALR: u32 = 0b000;
const FUNCT3_BRANCH_BEQ: u32 = 0b000;
const FUNCT3_LOAD_LD: u32 = 0b011;
const SHIFT_FUNCT3: u32 = 12;

// Opcode encodings
const RV_OP_LUI: u32 = 0b0110111;
const RV_OP_AUIPC: u32 = 0b0010111;
const RV_OP_JAL: u32 = 0b1101111;
const RV_OP_JALR: u32 = 0b1100111;
// Conditional Branches
const RV_OP_BRANCH: u32 = 0b1100011; // BEQ, BNE, BLT, BGE, BLTU, BGEU
const RV_OP_LOAD: u32 = 0b0000011; // LB, LH, LW, LBU, LHU
const RV_OP_STORE: u32 = 0b0100011; // SB, SH, SW
                                    // Integer Register-Immediate Instructions
const RV_OP_OPIMM: u32 = 0b0010011; // ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI
                                    // Integer Register Register Operations
const RV_OP_OP: u32 = 0b0110011; // ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND
const RV_OP_MISC_MEM: u32 = 0b0001111;
const RV_OP_SYSTEM: u32 = 0b1110011; // ECALL, EBREAK, CSR*

// 'rd' is register destination
fn decode_rd(ins: u32) -> usize {
    ((ins & 0b00000000000000000000111110000000) >> 7) as usize
}

// 'rs1' is register source 1
fn decode_rs1(ins: u32) -> usize {
    ((ins & 0b00000000000011111000000000000000) >> 15) as usize
}

// 'rs2' is register source 2
fn decode_rs2(ins: u32) -> usize {
    ((ins & ins & 0b00000001111100000000000000000000) >> 20) as usize
}

// 12 bits, sign-extended
fn decode_i_imm(ins: u32) -> i64 {
    (((ins & 0b11111111111100000000000000000000) >> 20) as i32) as i64
}

// 12 bits, sign-extended
fn decode_s_imm(ins: u32) -> i64 {
    ((((ins & 0b11111110000000000000000000000000) >> (25 - 5)) as i32)
        | ((ins & 0b00000000000000000000111110000000) >> 7) as i32) as i64
}
// 32 bits, sign-extended
fn decode_u_imm(ins: u32) -> i64 {
    ((ins & 0b11111111111111111111000000000000) as i32) as i64
}
// 12 bits, sign-extended
fn decode_b_imm(ins: u32) -> i64 {
    ((((ins & 0b10000000000000000000000000000000) >> (31 - 12)) as i32)
        | (((ins & 0b01111110000000000000000000000000) >> (25 - 5)) as i32)
        | (((ins & 0b00000000000000000000111100000000) >> (8 - 1)) as i32)
        | (((ins & 0b00000000000000000000000010000000) << -(7 - 11)) as i32)) as i64
}
// 32 bits, sign-extended
fn decode_j_imm(ins: u32) -> i64 {
    ((((ins & 0b10000000000000000000000000000000) >> (31 - 20)) as i32)
        | (((ins & 0b01111111111000000000000000000000) >> (21 - 1)) as i32)
        | (((ins & 0b00000000000100000000000000000000) >> (20 - 11)) as i32)
        | (((ins & 0b00000000000011111111000000000000) >> (12 - 12)) as i32)) as i64
}

pub fn decode(ins: u32) -> Instruction {
    use Opcode::*;
    match ins & MASK_OPCODE {
        RV_OP_LUI => {
            // U-Type
            let rd = decode_rd(ins);
            let imm = decode_u_imm(ins);
            return Instruction {
                op: Lui,
                rd,
                imm,
                ..Default::default()
            };
        }
        RV_OP_AUIPC => {
            // U-Type
            panic!("Unsupported Auipc instruction {:08x}", ins);
        }
        RV_OP_JAL => {
            // J-Type
            let rd = decode_rd(ins);
            let imm = decode_j_imm(ins);
            return Instruction {
                op: Jal,
                rd,
                imm,
                ..Default::default()
            };
        }
        RV_OP_JALR => {
            // I-Type
            let rd = decode_rd(ins);
            let rs1 = decode_rs1(ins);
            let imm = decode_i_imm(ins);
            // jalr
            match (ins & MASK_FUNCT3) >> SHIFT_FUNCT3 {
                FUNCT3_JALR => {
                    return Instruction {
                        op: Jalr,
                        rd,
                        rs1,
                        imm,
                        ..Default::default()
                    };
                }
                _ => panic!("Invalid Jalr instruction {:08x}", ins),
            }
        }
        RV_OP_BRANCH => {
            // B-Type
            let rs1 = decode_rs1(ins);
            let rs2 = decode_rs2(ins);
            let imm = decode_b_imm(ins);
            match (ins & MASK_FUNCT3) >> SHIFT_FUNCT3 {
                FUNCT3_BRANCH_BEQ => {
                    return Instruction {
                        op: Beq,
                        rs1,
                        rs2,
                        imm,
                        ..Default::default()
                    }
                }
                _ => panic!("Unsupported/Invalid Jalr instruction {:08x}", ins),
            }
        }
        RV_OP_LOAD => {
            // I-Type
            let rd = decode_rd(ins);
            let rs1 = decode_rs1(ins);
            let imm = decode_i_imm(ins);
            match (ins & MASK_FUNCT3) >> SHIFT_FUNCT3 {
                FUNCT3_LOAD_LD => {
                    return Instruction {
                        op: Ld,
                        rd,
                        rs1,
                        imm,
                        ..Default::default()
                    }
                }
                _ => panic!("Unsupported/Invalid Load instruction {:08x}", ins),
            }
        }
        RV_OP_STORE => {
            // S-Type
            let rs1 = decode_rs1(ins);
            let rs2 = decode_rs2(ins);
            let imm = decode_s_imm(ins);
            // sd
            panic!("Unsupported Store instruction {:08x}", ins);
        }
        RV_OP_OPIMM => {
            // I-Type
            let rd = decode_rd(ins);
            let rs1 = decode_rs1(ins);
            let imm = decode_i_imm(ins);
            // addi
            panic!("Unsupported Opimm instruction {:08x}", ins);
        }
        RV_OP_OP => {
            // R-Type
            let rd = decode_rd(ins);
            let rs1 = decode_rs1(ins);
            let rs2 = decode_rs2(ins);
            // add
            // sub
            // mul
            // divu
            // remu
            // sltu
            panic!("Unsupported Op instruction {:08x}", ins);
        }
        RV_OP_MISC_MEM => {
            panic!("Unsupported MiscMem instruction {:08x}", ins);
        }
        RV_OP_SYSTEM => {
            // I-Type
            let rd = decode_rd(ins);
            let rs1 = decode_rs1(ins);
            let imm = decode_i_imm(ins);
            // ecall
            panic!("Unsupported System instruction {:08x}", ins);
        }
        _ => {
            panic!("Invalid instruction {:08x}", ins);
        }
    }
}
