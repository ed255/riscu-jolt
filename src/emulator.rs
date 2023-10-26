//! Implementation of an emulator of RISC-U (RV64I subset)

pub mod decoder;
pub mod memory;

use crate::Registers;
use crate::{Instruction, Opcode, Step as GenericStep, VirtualOpcode};
use crate::{REG_SP, REG_V0};
use decoder::decode;
use memory::{Memory, MemoryTracer};

type Step<T> = GenericStep<T, Instruction>;

pub struct VirtPcMap {
    base: u64,
    map: Vec<u64>,
}

impl VirtPcMap {
    fn new(base: u64, len: u64) -> Self {
        Self {
            base,
            map: vec![0; len as usize],
        }
    }
    // return the vpc corresponding to pc
    fn get(&self, pc: u64) -> u64 {
        self.map[((pc - self.base) / 4) as usize]
    }
    // Set the mapping between pc and vpc
    fn set(&mut self, pc: u64, vpc: u64) {
        self.map[((pc - self.base) / 4) as usize] = vpc;
    }
}

pub struct Virt<T> {
    // Virtual PC.  This program counter points to instruction position in rom.
    pub pc: T,
    // Map from PC to Virtual PC
    pub vpc_map: VirtPcMap,
    // The rom has all the decoded instructions of the program, where some instructions have been
    // expanded using virtual instructions.
    pub rom: Vec<Instruction>,
    // Advice buffer used in virtual instructions.  This vector is filled when a multi-step
    // instruction is found with the advice values that subsequent ADVICE instructions will need.
    pub advices: Vec<T>,
    // Last advice used.  Instructions that consume an advice will store a copy here so that we
    // can later get it for the trace.
    pub last_advice: Option<T>,
}

impl<T: Default> Virt<T> {
    fn new<M: Memory>(mem: &mut M) -> Self {
        use Opcode::Virtual;
        use VirtualOpcode::*;
        let (text_addr, text_len) = mem.text();
        let mut rom = Vec::with_capacity((text_len / 4) as usize);
        let mut vpc = 0;
        let mut vpc_map = VirtPcMap::new(text_addr, text_len);
        for addr in (text_addr..(text_addr + text_len)).step_by(4) {
            vpc_map.set(addr, vpc);
            let inst_u32 = mem.fetch_u32(addr);
            let inst = decode(inst_u32).expect("Valid instruction");
            // NOTE: We replace the virtual `MOVE` instruction defined in Jolt by an Add with the
            // zero register, which is the canonical way to implement the `mv` pseudoinstruction in
            // RISC-V.  See Table 25.2 in The RISC-V Instruction Set Manual, Volume I: Unprivileged
            // ISA, Document Version 20191213
            let virt_insts = match inst.op {
                Opcode::Divu => {
                    let [v0, vq, vr, vqy] = [0, 1, 2, 3].map(|i| REG_V0 + i);
                    vec![
                        Instruction::new(Virtual(Advice), vq, 0, 0, 0),
                        Instruction::new(Virtual(Advice), vr, 0, 0, 0),
                        Instruction::new(Virtual(Mul), vqy, vq, inst.rs2, 0),
                        Instruction::new(Virtual(AssertLtu), 0, vr, inst.rs2, 0),
                        Instruction::new(Virtual(AssertLte), 0, vqy, inst.rs1, 0),
                        Instruction::new(Virtual(Add), v0, vqy, vr, 0),
                        Instruction::new(Virtual(AssertEq), 0, v0, inst.rs1, 0),
                        Instruction::new(Virtual(Add), inst.rd, vq, 0, 0),
                    ]
                }
                Opcode::Remu => {
                    let [v0, vq, vr, vqy] = [0, 1, 2, 3].map(|i| REG_V0 + i);
                    vec![
                        Instruction::new(Virtual(Advice), vq, 0, 0, 0),
                        Instruction::new(Virtual(Advice), vr, 0, 0, 0),
                        Instruction::new(Virtual(Mul), vqy, vq, inst.rs2, 0),
                        Instruction::new(Virtual(AssertLtu), 0, vr, inst.rs2, 0),
                        Instruction::new(Virtual(AssertLte), 0, vqy, inst.rs1, 0),
                        Instruction::new(Virtual(Add), v0, vqy, vr, 0),
                        Instruction::new(Virtual(AssertEq), 0, v0, inst.rs1, 0),
                        Instruction::new(Virtual(Add), inst.rd, vr, 0, 0),
                    ]
                }
                _ => {
                    rom.push(inst);
                    vpc += 1;
                    continue;
                }
            };
            for inst in virt_insts {
                rom.push(inst);
                vpc += 1;
            }
        }
        Virt {
            pc: T::default(),
            vpc_map,
            rom,
            advices: Vec::new(),
            last_advice: None,
        }
    }
}

#[derive(Default)]
pub struct Emulator<T, M: Memory> {
    pub pc: T,
    pub regs: Registers<T>,
    pub mem: M,
    // Enable Jolt virtual behaviour
    pub virt: Option<Virt<T>>,
}

impl<T: Default + From<u64>, M: Memory> Emulator<T, M> {
    pub fn new(mut mem: M, virt: bool) -> Self {
        let sp = mem.sp();
        let entry_point = mem.entry_point();
        let virt = if virt {
            Some(Virt::new(&mut mem))
        } else {
            None
        };
        let mut emu = Emulator {
            pc: T::default(),
            regs: Registers::default(),
            mem,
            virt,
        };
        emu.pc = entry_point.into();
        emu.regs[REG_SP] = sp.into();
        emu
    }
}

impl<T: Default + From<u64>, M: Memory> Emulator<T, MemoryTracer<M>> {
    pub fn new_tracer(mem: M, virt: bool) -> Self {
        let mem = MemoryTracer::new(mem);
        Self::new(mem, virt)
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
        self.pc += 4;
        if let Some(v) = self.virt.as_mut() {
            v.pc += 1;
        }
    }
    // #### Initialization

    // `lui rd,imm`: `rd = imm * 2^12; pc = pc + 4` with `-2^19 <= imm < 2^19`
    pub fn lui(&mut self, rd: usize, imm: i64) {
        debug_assert_eq!(imm & ((1 << 12) - 1), 0);
        self.regs[rd] = imm as u64;
        self.pc += 4;
        if let Some(v) = self.virt.as_mut() {
            v.pc += 1;
        }
    }
    // `addi rd,rs1,imm`: `rd = rs1 + imm; pc = pc + 4` with `-2^11 <= imm < 2^11`
    pub fn addi(&mut self, rd: usize, rs1: usize, imm: i64) {
        debug_assert!((-(1 << 11)..1 << 11).contains(&imm));
        let (result, _) = self.regs[rs1].overflowing_add(imm as u64);
        self.regs[rd] = result;
        self.pc += 4;
        if let Some(v) = self.virt.as_mut() {
            v.pc += 1;
        }
    }

    // #### Memory

    // `ld rd,imm(rs1)`: `rd = memory[rs1 + imm]; pc = pc + 4` with `-2^11 <= imm < 2^11`
    pub fn ld(&mut self, rd: usize, rs1: usize, imm: i64) {
        debug_assert!((-(1 << 11)..1 << 11).contains(&imm));
        let (addr, _) = self.regs[rs1].overflowing_add(imm as u64);
        self.regs[rd] = self.mem.read_u64(addr);
        self.pc += 4;
        if let Some(v) = self.virt.as_mut() {
            v.pc += 1;
        }
    }
    // `sd rs2,imm(rs1)`: `memory[rs1 + imm] = rs2; pc = pc + 4` with `-2^11 <= imm < 2^11`
    pub fn sd(&mut self, rs1: usize, rs2: usize, imm: i64) {
        debug_assert!((-(1 << 11)..1 << 11).contains(&imm));
        let (addr, _) = self.regs[rs1].overflowing_add(imm as u64);
        self.mem.write_u64(addr, self.regs[rs2]);
        self.pc += 4;
        if let Some(v) = self.virt.as_mut() {
            v.pc += 1;
        }
    }

    // #### Arithmetic

    // `add rd,rs1,rs2`: `rd = rs1 + rs2; pc = pc + 4`
    pub fn add(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let (result, _) = self.regs[rs1].overflowing_add(self.regs[rs2]);
        self.regs[rd] = result;
        self.pc += 4;
        if let Some(v) = self.virt.as_mut() {
            v.pc += 1;
        }
    }
    // `sub rd,rs1,rs2`: `rd = rs1 - rs2; pc = pc + 4`
    pub fn sub(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let (result, _) = self.regs[rs1].overflowing_sub(self.regs[rs2]);
        self.regs[rd] = result;
        self.pc += 4;
        if let Some(v) = self.virt.as_mut() {
            v.pc += 1;
        }
    }
    // `mul rd,rs1,rs2`: `rd = rs1 * rs2; pc = pc + 4`
    pub fn mul(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let (result, _) = self.regs[rs1].overflowing_mul(self.regs[rs2]);
        self.regs[rd] = result;
        self.pc += 4;
        if let Some(v) = self.virt.as_mut() {
            v.pc += 1;
        }
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
        self.pc += 4;
        if let Some(v) = self.virt.as_mut() {
            v.pc += 1;
        }
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
        self.pc += 4;
        if let Some(v) = self.virt.as_mut() {
            v.pc += 1;
        }
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
        self.pc += 4;
        if let Some(v) = self.virt.as_mut() {
            v.pc += 1;
        }
    }
    // #### Control

    // `beq rs1,rs2,imm`: `if (rs1 == rs2) { pc = pc + imm } else { pc = pc + 4 }` with `-2^12 <= imm < 2^12` and `imm % 2 == 0`
    pub fn beq(&mut self, rs1: usize, rs2: usize, imm: i64) {
        if self.regs[rs1] == self.regs[rs2] {
            debug_assert!((-(1 << 12)..1 << 12).contains(&imm));
            debug_assert_eq!(imm % 2, 0);
            assert_eq!(imm % 4, 0, "instruction-address-misaligned");
            self.pc = (self.pc as i64 + imm) as u64;
            if let Some(v) = self.virt.as_mut() {
                v.pc = v.vpc_map.get(self.pc);
            }
        } else {
            self.pc += 4;
            if let Some(v) = self.virt.as_mut() {
                v.pc += 1;
            }
        }
    }
    // `jal rd,imm`: `rd = pc + 4; pc = pc + imm` with `-2^20 <= imm < 2^20` and `imm % 2 == 0`
    pub fn jal(&mut self, rd: usize, imm: i64) {
        debug_assert!((-(1 << 20)..1 << 20).contains(&imm));
        debug_assert_eq!(imm % 2, 0);
        assert_eq!(imm % 4, 0, "instruction-address-misaligned");
        self.regs[rd] = self.pc + 4;
        self.pc = (self.pc as i64 + imm) as u64;
        if let Some(v) = self.virt.as_mut() {
            v.pc = v.vpc_map.get(self.pc);
        }
    }
    // `jalr rd,imm(rs1)`: `tmp = ((rs1 + imm) / 2) * 2; rd = pc + 4; pc = tmp` with `-2^11 <= imm < 2^11`
    pub fn jalr(&mut self, rd: usize, rs1: usize, imm: i64) {
        debug_assert!((-(1 << 12)..1 << 12).contains(&imm));
        let (tmp, _) = (self.regs[rs1] as i64).overflowing_add(imm);
        let tmp = (tmp as u64) & 0xfffffffffffffffe;
        assert_eq!(tmp % 4, 0, "instruction-address-misaligned");
        self.regs[rd] = self.pc + 4;
        self.pc = tmp;
        if let Some(v) = self.virt.as_mut() {
            v.pc = v.vpc_map.get(self.pc);
        }
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
        self.pc += 4;
        if let Some(v) = self.virt.as_mut() {
            v.pc += 1;
        }
    }

    // #### Jolt Virtual
    // NOTE: We use unwrap when updating the vpc to make sure the Virtual extension is enabled.
    // Running the emulator on Virtual instructions with the Virtual extension not enabled is
    // unsupported.

    pub fn v_assert_ltu(&mut self, rs1: usize, rs2: usize) {
        let v = self.virt.as_mut().expect("virtual extension");
        assert!(
            self.regs[rs1] < self.regs[rs2],
            "{} >= {}",
            self.regs[rs1],
            self.regs[rs2]
        );
        v.pc += 1;
    }
    pub fn v_assert_lte(&mut self, rs1: usize, rs2: usize) {
        let v = self.virt.as_mut().expect("virtual extension");
        assert!(
            self.regs[rs1] <= self.regs[rs2],
            "{} > {}",
            self.regs[rs1],
            self.regs[rs2]
        );
        v.pc += 1;
    }
    pub fn v_assert_eq(&mut self, rs1: usize, rs2: usize) {
        let v = self.virt.as_mut().expect("virtual extension");
        assert!(
            self.regs[rs1] == self.regs[rs2],
            "{} != {}",
            self.regs[rs1],
            self.regs[rs2]
        );
        v.pc += 1;
    }
    pub fn v_advice(&mut self, rd: usize) {
        let v = self.virt.as_mut().expect("virtual extension");
        // This vector is filled in the fetch step when virt is enabled.
        let advice = v.advices.pop().unwrap();
        v.last_advice = Some(advice);
        self.regs[rd] = advice;
        v.pc += 1;
    }
    pub fn v_add(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let v = self.virt.as_mut().expect("virtual extension");
        let (result, _) = self.regs[rs1].overflowing_add(self.regs[rs2]);
        self.regs[rd] = result;
        v.pc += 1;
    }
    pub fn v_mul(&mut self, rd: usize, rs1: usize, rs2: usize) {
        let v = self.virt.as_mut().expect("virtual extension");
        let (result, _) = self.regs[rs1].overflowing_mul(self.regs[rs2]);
        self.regs[rd] = result;
        v.pc += 1;
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
            Virtual(vop) => match vop {
                VirtualOpcode::AssertLtu => self.v_assert_ltu(rs1, rs2),
                VirtualOpcode::AssertLte => self.v_assert_lte(rs1, rs2),
                VirtualOpcode::AssertEq => self.v_assert_eq(rs1, rs2),
                VirtualOpcode::Advice => self.v_advice(rd),
                VirtualOpcode::Add => self.v_add(rd, rs1, rs2),
                VirtualOpcode::Mul => self.v_mul(rd, rs1, rs2),
            },
        }
    }
    pub fn step(&mut self) -> Instruction {
        use Opcode::*;
        // Fetch and decode original instruction
        let inst_u32 = self.mem.fetch_u32(self.pc);
        let inst = decode(inst_u32).expect("Valid instruction");
        let real_inst = if let Some(ref mut virt) = self.virt {
            // If we have a multi-step instruction that requires advice, fill the advice buffer
            // here.
            // When the current original instruction is multi-step, check wether we are at its
            // first vpc
            let vpc_pc_match = virt.vpc_map.get(self.pc) == virt.pc;
            if vpc_pc_match {
                match inst.op {
                    Divu | Remu => {
                        let (quotient, _) =
                            self.regs[inst.rs1].overflowing_div(self.regs[inst.rs2]);
                        let (remainder, _) =
                            self.regs[inst.rs1].overflowing_rem(self.regs[inst.rs2]);
                        virt.advices = vec![remainder, quotient];
                    }
                    _ => {}
                }
            }
            // Use the already decoded instruction at vpc (possibly virtual), which may not match
            // the original instruction
            virt.rom[virt.pc as usize].clone()
        } else {
            inst
        };
        self.exec(real_inst.clone());
        real_inst
    }
}

impl<M: Memory> Emulator<u64, MemoryTracer<M>> {
    pub fn exec_trace(&mut self, inst: Instruction) -> Step<u64> {
        let pc = self.pc;
        let vpc = self.virt.as_ref().map(|v| v.pc).unwrap_or(0);
        let regs = self.regs.clone();
        let mem_t = self.mem.t;
        self.exec(inst.clone());
        // For this method we skipped the instruction fetch, so all the memory ops come from
        // instruction execution.
        let mem_ops = self.mem.trace.clone();
        self.mem.trace.clear();
        let advice = self
            .virt
            .as_ref()
            .map(|v| v.last_advice.unwrap_or(0))
            .unwrap_or(0);
        Step {
            inst,
            pc,
            vpc,
            regs,
            mem_t,
            mem_ops,
            advice,
        }
    }
    pub fn step_trace(&mut self) -> Step<u64> {
        let pc = self.pc;
        let vpc = self.virt.as_ref().map(|v| v.pc).unwrap_or(0);
        let regs = self.regs.clone();
        let mem_t = self.mem.t;
        let inst = self.step();
        // The memory trace contains 4 entries for fetching the instruction (32 bits) and optionally more entries
        // from load / store instruction
        let mem_ops = self.mem.trace[4..].into();
        self.mem.trace.clear();
        let advice = self
            .virt
            .as_ref()
            .map(|v| v.last_advice.unwrap_or(0))
            .unwrap_or(0);
        Step {
            inst,
            pc,
            vpc,
            regs,
            mem_t,
            mem_ops,
            advice,
        }
    }
}
