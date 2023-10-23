//! Implementation of an emulator of RISC-U (RV64I subset)

pub mod decoder;
pub mod memory;

use crate::Registers;
use crate::REG_SP;
use crate::{Instruction, Opcode, Step as GenericStep};
use decoder::decode;
use memory::{Memory, MemoryTracer};

type Step<T> = GenericStep<T, Instruction>;

#[derive(Default)]
pub struct Emulator<T, M: Memory> {
    pub pc: T,
    pub regs: Registers<T>,
    pub mem: M,
}

impl<T: Default + From<u64>, M: Memory> Emulator<T, M> {
    pub fn new(mem: M) -> Self {
        let sp = mem.sp();
        let entry_point = mem.entry_point();
        let mut emu = Emulator {
            pc: T::default(),
            regs: Registers::default(),
            mem,
        };
        emu.pc = entry_point.into();
        emu.regs[REG_SP] = sp.into();
        emu
    }
}

impl<T: Default + From<u64>, M: Memory> Emulator<T, MemoryTracer<M>> {
    pub fn new_tracer(mem: M) -> Self {
        let mem = MemoryTracer::new(mem);
        Self::new(mem)
    }
}

const SYSCALL_EXIT: u64 = 93;
const SYSCALL_READ: u64 = 63;
const SYSCALL_WRITE: u64 = 64;
const SYSCALL_OPENAT: u64 = 56;
const SYSCALL_BRK: u64 = 214;

// Emulated instructions
impl<M: Memory> Emulator<u64, M> {
    pub fn nop(&mut self) {
        self.pc = self.pc + 4;
    }
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
    pub fn exec(&mut self, inst: Instruction) {
        use Opcode::*;
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
    pub fn step(&mut self) -> Instruction {
        // Fetch
        let inst_u32 = self.mem.fetch_u32(self.pc);
        // Decode
        let inst = decode(inst_u32).expect("Valid instruction");
        self.exec(inst.clone());
        inst
    }
}

impl<M: Memory> Emulator<u64, MemoryTracer<M>> {
    pub fn exec_trace(&mut self, inst: Instruction) -> Step<u64> {
        let pc = self.pc;
        let regs = self.regs.clone();
        let mem_t = self.mem.t;
        self.exec(inst.clone());
        // For this method we skipped the instruction fetch, so all the memory ops come from
        // instruction execution.
        let mem_ops = self.mem.trace.clone();
        self.mem.trace.clear();
        Step {
            inst,
            pc,
            regs,
            mem_t,
            mem_ops,
        }
    }
    pub fn step_trace(&mut self) -> Step<u64> {
        let pc = self.pc;
        let regs = self.regs.clone();
        let mem_t = self.mem.t;
        let inst = self.step();
        // The memory trace contains 4 entries for fetching the instruction (32 bits) and optionally more entries
        // from load / store instruction
        let mem_ops = self.mem.trace[4..].into();
        self.mem.trace.clear();
        Step {
            inst,
            pc,
            regs,
            mem_t,
            mem_ops,
        }
    }
}
