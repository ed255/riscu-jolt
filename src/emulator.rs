//! Implementation of an emulator of RISC-U (RV64I subset)

pub mod decoder;

use crate::Registers;
use crate::REG_SP;
use decoder::decode;

use elf::abi::PF_X;
use elf::endian::LittleEndian;
use elf::ElfBytes;

#[derive(Default)]
pub struct Emulator<T, M: Memory> {
    pub pc: T,
    pub regs: Registers<T>,
    pub mem: M,
}

impl<T: Default, M: Memory> Emulator<T, M> {
    pub fn new(mem: M) -> Self {
        Emulator {
            pc: T::default(),
            regs: Registers::default(),
            mem,
        }
    }
}

pub trait Memory {
    fn read_u8(&self, addr: u64) -> u8;

    fn read_u32(&self, addr: u64) -> u32 {
        let data = [0, 1, 2, 3].map(|i| self.read_u8(addr + i));
        u32::from_le_bytes(data)
    }

    fn read_u64(&self, addr: u64) -> u64 {
        let data = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| self.read_u8(addr + i));
        u64::from_le_bytes(data)
    }

    fn write_u8(&mut self, addr: u64, value: u8);

    fn write_u64(&mut self, addr: u64, value: u64) {
        let data = value.to_le_bytes();
        data.iter()
            .enumerate()
            .for_each(|(i, b)| self.write_u8(addr + i as u64, *b));
    }
}

#[derive(Default)]
pub struct NoMem;

impl Memory for NoMem {
    fn read_u8(&self, addr: u64) -> u8 {
        panic!("No memory")
    }

    fn write_u8(&mut self, addr: u64, value: u8) {
        panic!("No memory")
    }
}

// https://github.com/riscv-software-src/riscv-pk
pub struct RiscvPkMemoryMap(Vec<u8>);

const RISCV_PK_VMEM: u64 = 0x10000;
const RISCV_PK_ENTRY_POINT: u64 = RISCV_PK_VMEM;

impl RiscvPkMemoryMap {
    // Map address to RAM index
    fn map(&self, addr: u64) -> u64 {
        if addr < RISCV_PK_VMEM {
            panic!("Invalid low address 0x{:x} < 0x{:x}", addr, RISCV_PK_VMEM);
        } else if RISCV_PK_VMEM <= addr && addr < RISCV_PK_VMEM + self.0.len() as u64 {
            addr - RISCV_PK_VMEM
        } else {
            panic!(
                "Invalid high address 0x{:x} > 0x{:x}",
                addr,
                RISCV_PK_VMEM + self.0.len() as u64
            );
        }
    }
}

impl Memory for RiscvPkMemoryMap {
    fn read_u8(&self, addr: u64) -> u8 {
        self.0[self.map(addr) as usize]
    }

    fn write_u8(&mut self, addr: u64, value: u8) {
        let index = self.map(addr) as usize;
        self.0[index] = value;
    }
}

impl<T: Default + From<u64>> Emulator<T, RiscvPkMemoryMap> {
    fn new_pk(mem_size: usize, text: &[u8], data_addr: usize, data: &[u8]) -> Self {
        assert!(text.len() < mem_size);
        assert!((data_addr - RISCV_PK_ENTRY_POINT as usize) + data.len() < mem_size);
        let mut mem = RiscvPkMemoryMap(vec![0; mem_size]);
        // Load text section
        for (i, b) in text.iter().enumerate() {
            mem.write_u8(RISCV_PK_ENTRY_POINT + i as u64, text[i]);
        }
        // Load data section
        for (i, b) in data.iter().enumerate() {
            mem.write_u8((data_addr + i) as u64, data[i]);
        }
        let sp = RISCV_PK_ENTRY_POINT + mem_size as u64 - std::mem::size_of::<u64>() as u64;
        let mut emu = Self::new(mem);
        emu.pc = RISCV_PK_ENTRY_POINT.into();
        emu.regs[REG_SP] = sp.into();
        emu
    }

    pub fn new_pk_from_elf(mem_size: usize, file: &ElfBytes<LittleEndian>) -> Self {
        let entry_point = file.ehdr.e_entry;
        assert_eq!(file.ehdr.e_entry, RISCV_PK_ENTRY_POINT);
        let mut text_segment = None;
        let mut data_segment = None;
        let segments = file.segments().expect("Get segments");
        for segment in segments.iter() {
            if segment.p_flags & PF_X != 0 {
                if text_segment.is_some() {
                    panic!("Found 2 executable segments");
                }
                text_segment = Some(segment);
            } else {
                if data_segment.is_some() {
                    panic!("Found 2 data segments");
                }
                data_segment = Some(segment);
            }
        }
        match (text_segment, data_segment) {
            (Some(text_segment), Some(data_segment)) => {
                let text = file.segment_data(&text_segment).expect("Get text bytes");
                let data = file.segment_data(&data_segment).expect("Get data bytes");
                let data_addr = data_segment.p_vaddr;
                Self::new_pk(mem_size, text, data_addr as usize, data)
            }
            (None, Some(_)) => panic!("Missing text segment"),
            (Some(_), None) => panic!("Missing data segment"),
            (None, None) => panic!("Missing text & data segment"),
        }
    }
}

const SYSCALL_EXIT: u64 = 93;
const SYSCALL_READ: u64 = 63;
const SYSCALL_WRITE: u64 = 64;
const SYSCALL_OPENAT: u64 = 56;
const SYSCALL_BRK: u64 = 214;

// Emulated instructions
impl<M: Memory> Emulator<u64, M> {
    // #### Initialization

    // `lui rd,imm`: `rd = imm * 2^12; pc = pc + 4` with `-2^19 <= imm < 2^19`
    pub fn lui(&mut self, rd: usize, imm: i64) {
        debug_assert_eq!(imm & ((1 << 12) - 1), 0);
        self.regs[rd] = imm as u64;
        self.pc = self.pc + 4;
    }
    // `addi rd,rs1,imm`: `rd = rs1 + imm; pc = pc + 4` with `-2^11 <= imm < 2^11`
    pub fn addi(&mut self, rd: usize, rs1: usize, imm: i64) {
        debug_assert!(-(1 << 11) <= imm && imm < 1 << 11);
        let (result, _) = self.regs[rs1].overflowing_add(imm as u64);
        self.regs[rd] = result;
        self.pc = self.pc + 4;
    }

    // #### Memory

    // `ld rd,imm(rs1)`: `rd = memory[rs1 + imm]; pc = pc + 4` with `-2^11 <= imm < 2^11`
    pub fn ld(&mut self, rd: usize, rs1: usize, imm: i64) {
        debug_assert!(-(1 << 11) <= imm && imm < 1 << 11);
        let (addr, _) = self.regs[rs1].overflowing_add(imm as u64);
        println!(
            "DBG addr={:08x}, rs1={:08x}, imm={}",
            addr, self.regs[rs1], imm
        );
        self.regs[rd] = self.mem.read_u64(addr);
        self.pc = self.pc + 4;
    }
    // `sd rs2,imm(rs1)`: `memory[rs1 + imm] = rs2; pc = pc + 4` with `-2^11 <= imm < 2^11`
    pub fn sd(&mut self, rs1: usize, rs2: usize, imm: i64) {
        debug_assert!(-(1 << 11) <= imm && imm < 1 << 11);
        let (addr, _) = self.regs[rs1].overflowing_add(imm as u64);
        self.mem.write_u64(addr, self.regs[rs2]);
        self.pc = self.pc + 4;
    }

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
            debug_assert!(-(1 << 12) <= imm && imm < 1 << 12);
            debug_assert_eq!(imm % 2, 0);
            assert_eq!(imm % 4, 0, "instruction-address-misaligned");
            self.pc = (self.pc as i64 + imm as i64) as u64;
        } else {
            self.pc = self.pc + 4;
        }
    }
    // `jal rd,imm`: `rd = pc + 4; pc = pc + imm` with `-2^20 <= imm < 2^20` and `imm % 2 == 0`
    pub fn jal(&mut self, rd: usize, imm: i64) {
        debug_assert!(-(1 << 20) <= imm && imm < 1 << 20);
        debug_assert_eq!(imm % 2, 0);
        assert_eq!(imm % 4, 0, "instruction-address-misaligned");
        self.regs[rd] = self.pc + 4;
        self.pc = (self.pc as i64 + imm as i64) as u64;
    }
    // `jalr rd,imm(rs1)`: `tmp = ((rs1 + imm) / 2) * 2; rd = pc + 4; pc = tmp` with `-2^11 <= imm < 2^11`
    pub fn jalr(&mut self, rd: usize, rs1: usize, imm: i64) {
        debug_assert!(-(1 << 12) <= imm && imm < 1 << 12);
        let (tmp, _) = (self.regs[rs1] as i64).overflowing_add(imm);
        let tmp = (tmp as u64) & 0xfffffffffffffffe;
        assert_eq!(tmp % 4, 0, "instruction-address-misaligned");
        self.regs[rd] = self.pc + 4;
        self.pc = tmp;
    }

    // #### System

    // `ecall`: system call number is in `a7`, actual parameters are in `a0-a3`, return value is in `a0`.
    pub fn ecall(&mut self) {
        let ecall_num = self.regs[17];
        match ecall_num {
            SYSCALL_BRK => {
                println!("DBG: Skipping unimplemented ecall BRK ");
            }
            SYSCALL_EXIT => {
                panic!(
                    "Exit ecall with arguments {:?}",
                    [self.regs[10], self.regs[11], self.regs[12], self.regs[13]]
                );
            }
            _ => {
                panic!("Unimplemented ecall {}", self.regs[17]);
            }
        }
        self.pc = self.pc + 4;
    }
}

impl<M: Memory> Emulator<u64, M> {
    pub fn step(&mut self) {
        use Opcode::*;
        // Fetch
        let inst_u32 = self.mem.read_u32(self.pc);
        // Decode
        let inst = decode(inst_u32).expect("Valid instruction");
        let Instruction {
            op,
            rd,
            rs1,
            rs2,
            imm,
        } = inst;
        // Execute
        match op {
            Lui => self.lui(rd, imm),
            Addi => self.addi(rd, rs1, imm),
            Ld => self.ld(rd, rs1, imm),
            Sd => self.sd(rs1, rs2, imm),
            Add => self.add(rd, rs1, rs2),
            Sub => self.sub(rd, rs1, rs2),
            Mul => self.mul(rd, rs1, rs2),
            Divu => self.divu(rd, rs1, rs2),
            Remu => self.remu(rd, rs1, rs2),
            Sltu => self.sltu(rd, rs1, rs2),
            Beq => self.beq(rs1, rs2, imm),
            Jal => self.jal(rd, imm),
            Jalr => self.jalr(rd, rs1, imm),
            Ecall => self.ecall(),
        }
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
pub struct Instruction {
    op: Opcode,
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_new_pk_from_elf() {
        let path = std::path::PathBuf::from("riscu_examples/c/fibo.bin");
        let file_data = std::fs::read(path).expect("Could not read file.");
        let slice = file_data.as_slice();
        let file = ElfBytes::<LittleEndian>::minimal_parse(slice).expect("Open test1");
        let emu = Emulator::<u64, _>::new_pk_from_elf(1024 * 1024, &file);
    }
}
