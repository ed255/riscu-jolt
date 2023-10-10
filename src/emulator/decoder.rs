use super::*;

const MASK_OPCODE: u32 = 0b00000000000000000000000001111111;
const MASK_FUNCT3: u32 = 0b00000000000000000111000000000000;
const MASK_FUNCT7: u32 = 0b11111110000000000000000000000000;
const MASK_FUNCT12: u32 = 0b11111111111100000000000000000000;
const MASK_IMM_HI: u32 = 0b11111110000000000000000000000000;
const FUNCT3_JALR: u32 = 0b000;
const FUNCT3_BRANCH_BEQ: u32 = 0b000;
const FUNCT3_LOAD_LD: u32 = 0b011;
const FUNCT3_STORE_SD: u32 = 0b011;
const FUNCT3_OPIMM_ADDI: u32 = 0b000;
const FUNCT3_OP_ADDSUB: u32 = 0b000;
const FUNCT3_OP_SLTU: u32 = 0b011;
const FUNCT3_OP_MUL: u32 = 0b000;
const FUNCT3_OP_DIVU: u32 = 0b101;
const FUNCT3_OP_REMU: u32 = 0b111;
const FUNCT3_SYSTEM_PRIV: u32 = 0b000;
const FUNCT7_OP_ADD: u32 = 0b0000000;
const FUNCT7_OP_SUB: u32 = 0b0100000;
const FUNCT7_OP_I: u32 = 0b0000000;
const FUNCT7_OP_M: u32 = 0b0000001;
const FUNCT12_PRIV_ECALL: u32 = 0b000000000000;
const SHIFT_FUNCT3: u32 = 12;
const SHIFT_FUNCT7: u32 = 25;
const SHIFT_FUNCT12: u32 = 20;
const SHIFT_IMM_HI: u32 = 25;

// Opcode encodings
const RV_OP_LUI: u32 = 0b0110111;
const RV_OP_AUIPC: u32 = 0b0010111;
const RV_OP_JAL: u32 = 0b1101111;
const RV_OP_JALR: u32 = 0b1100111;
// Conditional Branches
const RV_OP_BRANCH: u32 = 0b1100011; // BEQ, BNE, BLT, BGE, BLTU, BGEU
const RV_OP_LOAD: u32 = 0b0000011; // LB, LH, LW, LBU, LHU, LWU, LD
const RV_OP_STORE: u32 = 0b0100011; // SB, SH, SW, SD
                                    // Integer Register-Immediate Instructions
const RV_OP_OPIMM: u32 = 0b0010011; // ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI, SLLI,
                                    // SRLI, SRAI
                                    // Integer Register Register Operations
const RV_OP_OP: u32 = 0b0110011; // ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND, ADDIW, SLLIW,
                                 // SRLIW, SRAIW, ADDW, SUBW, SLLW, SRLW, SRAW
                                 // (RV32M) MUL, MULH, MULHSU, MULHU, DIV, DIVU, REM, REMU
const RV_OP_RV64M: u32 = 0b0111011; // MULW, DIVW, DIVUW, REMW, REMUW
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
    ((ins & 0b11111111111100000000000000000000) as i32 >> 20) as i64
}

// 12 bits, sign-extended
fn decode_s_imm(ins: u32) -> i64 {
    (((ins & 0b11111110000000000000000000000000) as i32 >> (25 - 5))
        | ((ins & 0b00000000000000000000111110000000) as i32 >> 7)) as i64
}
// 32 bits, sign-extended
fn decode_u_imm(ins: u32) -> i64 {
    ((ins & 0b11111111111111111111000000000000) as i32) as i64
}
// 12 bits, sign-extended
fn decode_b_imm(ins: u32) -> i64 {
    (((ins & 0b10000000000000000000000000000000) as i32 >> (31 - 12))
        | ((ins & 0b01111110000000000000000000000000) as i32 >> (25 - 5))
        | ((ins & 0b00000000000000000000111100000000) as i32 >> (8 - 1))
        | (((ins & 0b00000000000000000000000010000000) as i32) << -(7 - 11))) as i64
}
// 32 bits, sign-extended
fn decode_j_imm(ins: u32) -> i64 {
    (((ins & 0b10000000000000000000000000000000) as i32 >> (31 - 20))
        | ((ins & 0b01111111111000000000000000000000) as i32 >> (21 - 1))
        | ((ins & 0b00000000000100000000000000000000) as i32 >> (20 - 11))
        | ((ins & 0b00000000000011111111000000000000) as i32 >> (12 - 12))) as i64
}

#[derive(Debug)]
pub struct InvalidInstr(pub u32);

pub fn decode(ins: u32) -> Result<Instruction, InvalidInstr> {
    use Opcode::*;
    let rd = decode_rd(ins);
    let rs1 = decode_rs1(ins);
    let rs2 = decode_rs2(ins);
    let funct12 = (ins & MASK_FUNCT12) >> SHIFT_FUNCT12;
    let funct7 = (ins & MASK_FUNCT7) >> SHIFT_FUNCT7;
    let funct3 = (ins & MASK_FUNCT3) >> SHIFT_FUNCT3;
    Ok(match ins & MASK_OPCODE {
        // U-Type
        RV_OP_LUI => {
            let imm = decode_u_imm(ins);
            Instruction::new(Lui, rd, 0, 0, imm)
        }
        // U-Type
        RV_OP_AUIPC => {
            return Err(InvalidInstr(ins));
        }
        // J-Type
        RV_OP_JAL => {
            let imm = decode_j_imm(ins);
            Instruction::new(Jal, rd, 0, 0, imm)
        }
        // I-Type
        RV_OP_JALR => {
            let imm = decode_i_imm(ins);
            match funct3 {
                FUNCT3_JALR => Instruction::new(Jalr, rd, rs1, 0, imm),
                _ => return Err(InvalidInstr(ins)),
            }
        }
        // B-Type
        RV_OP_BRANCH => {
            let imm = decode_b_imm(ins);
            match funct3 {
                FUNCT3_BRANCH_BEQ => Instruction::new(Beq, 0, rs1, rs2, imm),
                _ => return Err(InvalidInstr(ins)),
            }
        }
        // I-Type
        RV_OP_LOAD => {
            let imm = decode_i_imm(ins);
            match funct3 {
                FUNCT3_LOAD_LD => Instruction::new(Ld, rd, rs1, 0, imm),
                _ => return Err(InvalidInstr(ins)),
            }
        }
        // S-Type
        RV_OP_STORE => {
            let imm = decode_s_imm(ins);
            match funct3 {
                FUNCT3_STORE_SD => Instruction::new(Sd, 0, rs1, rs2, imm),
                _ => return Err(InvalidInstr(ins)),
            }
        }
        // I-Type
        RV_OP_OPIMM => {
            let imm = decode_i_imm(ins);
            match funct3 {
                FUNCT3_OPIMM_ADDI => Instruction::new(Addi, rd, rs1, 0, imm),
                _ => return Err(InvalidInstr(ins)),
            }
        }
        // R-Type
        RV_OP_OP => match (funct7, funct3) {
            (FUNCT7_OP_ADD, FUNCT3_OP_ADDSUB) => Instruction::new(Add, rd, rs1, rs2, 0),
            (FUNCT7_OP_SUB, FUNCT3_OP_ADDSUB) => Instruction::new(Sub, rd, rs1, rs2, 0),
            (FUNCT7_OP_I, FUNCT3_OP_SLTU) => Instruction::new(Sltu, rd, rs1, rs2, 0),
            (FUNCT7_OP_M, FUNCT3_OP_MUL) => Instruction::new(Mul, rd, rs1, rs2, 0),
            (FUNCT7_OP_M, FUNCT3_OP_DIVU) => Instruction::new(Divu, rd, rs1, rs2, 0),
            (FUNCT7_OP_M, FUNCT3_OP_REMU) => Instruction::new(Remu, rd, rs1, rs2, 0),
            _ => return Err(InvalidInstr(ins)),
        },
        RV_OP_MISC_MEM => {
            return Err(InvalidInstr(ins));
        }
        // I-Type
        RV_OP_SYSTEM => {
            let imm = decode_i_imm(ins);
            match (funct12, funct3) {
                (FUNCT12_PRIV_ECALL, FUNCT3_SYSTEM_PRIV) => {
                    if rd != 0 || rs1 != 0 {
                        return Err(InvalidInstr(ins));
                    }
                    Instruction::new(Ecall, 0, 0, 0, 0)
                }
                _ => return Err(InvalidInstr(ins)),
            }
        }
        _ => return Err(InvalidInstr(ins)),
    })
}

#[cfg(test)]
mod test {
    use super::*;
    use elf::abi::PF_X;
    use elf::endian::LittleEndian;
    use elf::ElfBytes;

    // This test is ignored because it requires an external compiled binary
    #[test]
    #[ignore]
    fn test_decoder() {
        let path = std::path::PathBuf::from("riscu_examples/c/fibo.bin");
        let file_data = std::fs::read(path).expect("Could not read file.");
        let slice = file_data.as_slice();
        let file = ElfBytes::<LittleEndian>::minimal_parse(slice).expect("Open test1");
        let entry_point = file.ehdr.e_entry;
        println!("Entry: 0x{:x}", entry_point);
        let segments = file.segments().expect("Get segments");
        for segment in segments.iter() {
            if segment.p_flags & PF_X != 0 {
                println!("{:?}", segment);
                let data = file.segment_data(&segment).expect("Get segment data");
                for bytes in data.chunks(4) {
                    let ins_value = u32::from_le_bytes(bytes.try_into().unwrap());
                    let ins = decode(ins_value);
                    println!("{:?}", ins);
                }
            }
        }
    }
}
