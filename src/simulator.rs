//! Implementation of a zk circuit simulator of RISC-U (RV64I subset) following the Jolt approach:
//! implementing the verification of each instruction via Lasso decomposable tables with some small
//! arithmetic constraints.

use crate::expr::Arithmetic;
use crate::expr::Var;
use crate::Registers;
use crate::{Instruction, Opcode, Step as GenericStep, RW};

use ark_ff::{biginteger::BigInteger, PrimeField};
use std::fmt::{self, Display};
use std::marker::PhantomData;

type Step<F> = GenericStep<F, JoltInstruction>;

trait ToLeBits {
    type T;

    fn to_le_bits(&self) -> Vec<Self::T>;
}

impl<F: PrimeField> ToLeBits for F {
    type T = F;
    fn to_le_bits(&self) -> Vec<Self::T> {
        self.into_bigint()
            .to_bits_le()
            .iter()
            .map(|bit| F::from(*bit))
            .collect()
    }
}

/// Variable type for MLE expressions
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Bit {
    pub var: usize,
    pub index: usize,
}

impl Var for Bit {}

const VARS: &str = "xyzabcdefghijklmnopqrstuvw";

impl Display for Bit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        assert!(self.var < VARS.len());
        write!(f, "{}_{}", &VARS[self.var..self.var + 1], self.index)
    }
}

/// Variable type for Combine Lookup expressions
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TableEval {
    pub table: &'static str,
    pub chunk: usize,
}

impl Var for TableEval {}

impl Display for TableEval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[r_{}]", self.table, self.chunk)
    }
}

#[derive(Default)]
struct LookupTables<F: PrimeField> {
    _marker: PhantomData<F>,
}

fn f_pow<F: Arithmetic>(base: usize, exp: usize) -> F {
    let (result, overflow) = (base as u128).overflowing_pow(exp as u32);
    assert!(!overflow);
    F::from(result)
}

#[derive(Default)]
pub struct SubTableMLE<F: Arithmetic> {
    _marker: PhantomData<F>,
}

impl<F: Arithmetic> SubTableMLE<F> {
    // Equalty between two n bits inputs.  Outputs 1 if x == y, 0 otherwise.
    // SubTable EQ
    // Ref Jolt 4.2.1 (4)
    pub fn eq(x: &[F], y: &[F]) -> F {
        assert_eq!(x.len(), y.len());
        let mut result = F::one();
        for i in 0..x.len() {
            result *= x[i].clone() * &y[i] + (F::one() - &x[i]) * (F::one() - &y[i]);
        }
        result
    }

    // Return the input (identity)
    fn range_full(x: &[F]) -> F {
        let mut result = F::zero();
        for i in 0..x.len() {
            result += x[i].clone() * f_pow::<F>(2, i);
        }
        result
    }

    // Return the input if it uses up to cuttoff bits, or 0 otherwise.
    fn range_remainder(cutoff: usize, x: &[F]) -> F {
        let mut result = F::zero();
        for i in 0..x.len() {
            if i < cutoff {
                result += x[i].clone() * f_pow::<F>(2, i);
            } else {
                result *= F::one() - x[i].clone();
            }
        }
        result
    }

    // Return always 0.
    fn range_zeros(x: &[F]) -> F {
        F::zero()
    }

    // Return the input with the bits after cuttoff bit set to 0.
    fn range_zero_upper(cutoff: usize, x: &[F]) -> F {
        let mut result = F::zero();
        for i in 0..x.len() {
            if i < cutoff {
                result += x[i].clone() * f_pow::<F>(2, i);
            }
        }
        result
    }

    // Return the input with the less significant bit set to 0.
    fn zero_first_bit(x: &[F]) -> F {
        let mut result = F::zero();
        for i in 1..x.len() {
            result += x[i].clone() * f_pow::<F>(2, i);
        }
        result
    }

    // Ref Jolt 4.2.2 (5)
    fn ltu_i(i: usize, x: &[F], y: &[F]) -> F {
        assert_eq!(x.len(), y.len());
        (F::one() - &x[i]) * &y[i] * Self::eq(&x[i + 1..], &y[i + 1..])
    }

    // Lower than between two n bits inputs.  Outputs 1 if x < y, 0 otherwise.
    // SubTable LTU
    // Ref Jolt 4.2.2 (6)
    fn ltu(x: &[F], y: &[F]) -> F {
        assert_eq!(x.len(), y.len());
        let mut result = F::zero();
        for i in 0..x.len() {
            result += Self::ltu_i(i, x, y);
        }
        result
    }
}

#[derive(Default)]
pub struct CombineLookups<F: Arithmetic> {
    _marker: PhantomData<F>,
}

// evals is [T_1[r_1], ..., T_k[r_1],
//           T_{k+1}[r_2], ..., T_{2k}[r_2],
//                      ...,
//           T_{a-k+1}[r_c], ..., T_a[r_c]]
// These methods are the `g` expressions to combine the evaluations of the subtables into a lookup
// result.
impl<F: Arithmetic> CombineLookups<F> {
    // Ref Jolt 4.2.2
    // LowerThan using LTU and EQ SubTables
    pub fn ltu(evals: &[F]) -> F {
        let c = evals.len() / 2;
        let evals_ltu = |i| &evals[i * 2];
        let evals_eq = |i| &evals[i * 2 + 1];
        let mut result = F::zero();
        let mut eq_acc = F::one();
        // Errata: Jolt defines this loop from 0 to C, but that means: this chunk is LT and the
        // previous chunks are EQ.  We're in little endian, so that condition doesn't represent the
        // LT condition.  Instead it should be: this chunk is LT and the following chunks are EQ.
        // Hence the loop from C to 0.
        for i in (0..c).rev() {
            result += evals_ltu(i).clone() * eq_acc.clone();
            eq_acc *= evals_eq(i).clone();
        }
        result
    }
    // Ref Jolt 4.2.1
    // Equality using EQ SubTable
    pub fn eq(evals: &[F]) -> F {
        let c = evals.len() / 1;
        let evals_eq = |i| &evals[i];
        let mut result = F::one();
        for i in 0..c {
            result *= evals_eq(i).clone();
        }
        result
    }

    // Range using doing linear combination of the table evaluations.  Can be used for:
    // - range check with range_full, range_remainder, range_zeros SubTables
    // - zero upper bits with range_full, range_zero_upper, range_zeros SubTables
    // - zero upper bits & zero lower bit with zero_first_bit, range_full, range_zero_upper
    //   SubTables
    pub fn range(chunk_len: usize, evals: &[F]) -> F {
        let c = evals.len() / 1;
        let mut result = F::zero();
        let base = 2usize.pow(chunk_len as u32);
        for i in 0..c {
            result += evals[i].clone() * f_pow::<F>(base, i);
        }
        result
    }
}

// Notation:
// - c: number of chunks
// - m: subtable size
// - W: word size (32 for RV32, 64 for RV64)
impl<F: PrimeField> LookupTables<F> {
    // Take an input of `scr_bits_len` bits and return the lowest `n_bits` bits.  THis effectively
    // sets to 0 any bit that appears after the first `n_bits` bits.  This can be used to remove
    // the overflow bit after an addition.
    // From this we can build FullTable W+1_to_W, W*2_to_W, etc.
    fn zero_upper_bits(src_bits_len: usize, n_bits: usize, chunk_len: usize, src: F) -> F {
        assert!(src.into_bigint().num_bits() as usize <= src_bits_len);
        let src_bits = &src.to_le_bits()[..src_bits_len];
        // NOTE: If cutoff = 0, then we only need `range_full` and `range_zeros` subtables.
        let cutoff = n_bits % chunk_len;
        let mut evals = vec![F::ZERO; src_bits_len.div_ceil(chunk_len)];
        for (i, src_i) in src_bits.chunks(chunk_len).enumerate() {
            if i * chunk_len + chunk_len <= n_bits {
                // n_bits point lands in a higher chunk
                evals[i] = SubTableMLE::range_full(src_i);
            } else if i * chunk_len < n_bits && n_bits < (i + 1) * chunk_len {
                // n_bits point lands in this chunk
                evals[i] = SubTableMLE::range_zero_upper(cutoff, src_i);
            } else {
                // n_bits point lands in a lower chunk
                evals[i] = SubTableMLE::range_zeros(src_i);
            }
        }
        CombineLookups::range(chunk_len, &evals)
    }
    // Take an input of `scr_bits_len` bits and return the lowest `n_bits` bits, with the 0th bit set to 0.
    fn zero_upper_bits_lower_bit(
        src_bits_len: usize,
        n_bits: usize,
        chunk_len: usize,
        src: F,
    ) -> F {
        assert!(src.into_bigint().num_bits() as usize <= src_bits_len);
        // Sanity check to make sure that the chunk where we set the 0th bit to 0 and the chunk
        // where we do range_zero_upper are not the same.
        assert!(chunk_len <= src_bits_len);
        let src_bits = &src.to_le_bits()[..src_bits_len];
        let cutoff = n_bits % chunk_len;
        let mut evals = vec![F::ZERO; src_bits_len.div_ceil(chunk_len)];
        for (i, src_i) in src_bits.chunks(chunk_len).enumerate() {
            if i == 0 {
                evals[i] = SubTableMLE::zero_first_bit(src_i);
            } else if i * chunk_len + chunk_len <= n_bits {
                // n_bits point lands in a higher chunk
                evals[i] = SubTableMLE::range_full(src_i);
            } else if i * chunk_len < n_bits && n_bits < (i + 1) * chunk_len {
                // n_bits point lands in this chunk
                evals[i] = SubTableMLE::range_zero_upper(cutoff, src_i);
            } else {
                // n_bits point lands in a lower chunk
                evals[i] = SubTableMLE::range_zeros(src_i);
            }
        }
        CombineLookups::range(chunk_len, &evals)
    }
    // Range behaves like the identity from 0 to 2^`n_bits` and as the zero function onwards
    fn range_nbits(src_bits_len: usize, n_bits: usize, c: usize, src: F) -> F {
        assert!(src.into_bigint().num_bits() as usize <= src_bits_len);
        let chunk_len = src_bits_len / c;
        let src_bits = src.to_le_bits();
        // NOTE: If cutoff = 0, then we only need `range_full` and `range_zeros` subtables.
        let cutoff = n_bits % (chunk_len);
        let mut evals = vec![F::ZERO; 1 * c];
        for i in 0..c {
            let src_i = &src_bits[i * chunk_len..(i + 1) * chunk_len];
            if i * chunk_len + chunk_len <= n_bits {
                // n_bits point lands in a higher chunk
                evals[i] = SubTableMLE::range_full(src_i);
            } else if i * chunk_len < n_bits && n_bits < (i + 1) * chunk_len {
                // n_bits point lands in this chunk
                evals[i] = SubTableMLE::range_remainder(cutoff, src_i);
            } else {
                // n_bits point lands in a lower chunk
                evals[i] = SubTableMLE::range_zeros(src_i);
            }
        }
        CombineLookups::range(chunk_len, &evals)
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
        let mut evals = vec![F::ZERO; 1 * c];
        for i in 0..c {
            let x_i = &x[i * (w / c)..(i + 1) * (w / c)];
            let y_i = &y[i * (w / c)..(i + 1) * (w / c)];
            evals[i] = SubTableMLE::eq(x_i, y_i);
        }
        CombineLookups::eq(&evals)
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
        let mut evals = vec![F::ZERO; 2 * c];
        for i in 0..c {
            let x_i = &x[i * (w / c)..(i + 1) * (w / c)];
            let y_i = &y[i * (w / c)..(i + 1) * (w / c)];
            evals[i * 2] = SubTableMLE::ltu(x_i, y_i);
            evals[i * 2 + 1] = SubTableMLE::eq(x_i, y_i);
        }
        CombineLookups::ltu(&evals)
    }
}

#[derive(Debug, Clone, Default)]
pub struct OpFlags {
    first_rs2: bool,
    second_imm: bool,
    // load from memory
    load: bool,
    store: bool,
    jump: bool,
    branch: bool,
    update_rd: bool,
    add: bool,
    sub: bool,
    mult: bool,
    non_deterministic: bool,
    is_assertion_false: bool,
    is_assertion_true: bool,
    is_positive: bool,
}

#[derive(Debug, Clone)]
pub struct JoltInstruction {
    pub op: Opcode,
    rd: usize,
    rs1: usize,
    rs2: usize,
    imm: u64,
    opflags: OpFlags,
}

impl From<Instruction> for JoltInstruction {
    fn from(inst: Instruction) -> Self {
        let mut opflags = OpFlags::default();
        match inst.op {
            Opcode::Lui => {
                opflags.second_imm = true;
                opflags.update_rd = true;
            }
            Opcode::Addi => {
                opflags.second_imm = true;
                opflags.add = true;
                opflags.update_rd = true;
            }
            Opcode::Ld => {
                opflags.load = true;
                opflags.update_rd = true;
                opflags.add = true;
            }
            Opcode::Sd => {
                opflags.store = true;
                opflags.add = true;
            }
            Opcode::Add => {
                opflags.add = true;
                opflags.update_rd = true;
            }
            Opcode::Sub => {
                opflags.sub = true;
                opflags.update_rd = true;
            }
            Opcode::Mul => {
                opflags.mult = true;
                opflags.update_rd = true;
            }
            Opcode::Divu => {
                // DIVU is a virtual instruction
                // uses ADVICE, MUL, ASSERT, ADD, MOVE
                // According to the Jolt 6.3's virtual sequence
                opflags.non_deterministic = true;
                opflags.mult = true;
                opflags.is_assertion_true = true;
                opflags.add = true;
                opflags.update_rd = true;
            }
            Opcode::Remu => {
                // Remu is a virtual instruction
                // uses ADVICE, MUL, ASSERT, ADD, MOVE
                // According to the Jolt 6.3's virtual sequence
                opflags.non_deterministic = true;
                opflags.mult = true;
                opflags.is_assertion_true = true;
                opflags.add = true;
                opflags.update_rd = true;
            }
            Opcode::Sltu => {
                opflags.update_rd = true;
            }
            Opcode::Beq => {
                opflags.branch = true;
                opflags.is_positive = inst.imm >= 0;
            }
            Opcode::Jal => {
                opflags.jump = true;
                opflags.is_positive = inst.imm >= 0;
            }
            Opcode::Jalr => {
                opflags.jump = true;
                opflags.is_positive = inst.imm >= 0;
            }
            Opcode::Ecall => {}
            Opcode::Virtual(vop) => todo!(),
        };
        let imm = if opflags.branch & !opflags.is_positive {
            inst.imm.unsigned_abs()
        } else {
            inst.imm as u64
        };
        JoltInstruction {
            op: inst.op,
            rd: inst.rd,
            rs1: inst.rs1,
            rs2: inst.rs2,
            imm,
            opflags,
        }
    }
}

impl<F: From<u64>> From<GenericStep<u64, Instruction>> for GenericStep<F, JoltInstruction> {
    fn from(step: GenericStep<u64, Instruction>) -> Self {
        Step {
            pc: F::from(step.pc),
            vpc: F::from(step.vpc),
            inst: step.inst.into(),
            regs: step.regs.into_f(),
            mem_ops: step.mem_ops,
            mem_t: step.mem_t,
            advice: step.advice,
        }
    }
}

#[derive(Default)]
pub struct Simulator<F: PrimeField> {
    pub(crate) pc: F,
    pub(crate) regs: Registers<F>,
    // TODO: Memory simulation
    // TODO: W
    // TODO: C
}

// TODO, move these two constants to parameters of the object.
const W: usize = 64;
const C: usize = 4;

fn neg<F: Arithmetic>(x: F) -> F {
    F::one() - x
}

// Simulated zk circuit instructions with Lasso lookups
impl<F: PrimeField> Simulator<F> {
    // #### Initialization

    // `lui rd,imm`: `rd = imm * 2^12; pc = pc + 4` with `-2^19 <= imm < 2^19`
    pub fn t_lui(step: &Step<F>, step_next: &Step<F>) {
        let inst = &step.inst;
        // Ref: Jolt 5.5
        // No lookup required
        assert_eq!(step_next.regs[inst.rd], F::from(inst.imm));
        assert_eq!(step_next.pc, step.pc + F::from(4u32));
        assert_eq!(step.mem_ops.len(), 0);
        assert_eq!(step_next.mem_t, step.mem_t);
    }
    // `addi rd,rs1,imm`: `rd = rs1 + imm; pc = pc + 4` with `-2^11 <= imm < 2^11`
    pub fn t_addi(step: &Step<F>, step_next: &Step<F>) {
        let inst = &step.inst;
        // Ref: Jolt 5.2 (Same as ADD)
        // Index. z = x + y over the native field
        let z = step.regs[inst.rs1] + F::from(inst.imm);
        // Lookup. z has W+1 bits.  Take lowest W bits via lookup table
        let result = LookupTables::zero_upper_bits(W + 1, W, W / C, z);
        assert_eq!(step_next.regs[inst.rd], result);
        assert_eq!(step_next.pc, step.pc + F::from(4u32));
        assert_eq!(step.mem_ops.len(), 0);
        assert_eq!(step_next.mem_t, step.mem_t);
    }

    // #### Memory

    // `ld rd,imm(rs1)`: `rd = memory[rs1 + imm]; pc = pc + 4` with `-2^11 <= imm < 2^11`
    pub fn t_ld(step: &Step<F>, step_next: &Step<F>) {
        let inst = &step.inst;
        // Ref: Jolt 5.8 (Sum is done as in Branch instructions)
        // Index. z = x + y over the native field
        let z = step.regs[inst.rs1] + F::from(inst.imm);
        // Lookup. z has W+1 bits.  Take lowest W bits via lookup table
        let addr = LookupTables::zero_upper_bits(W + 1, W, W / C, z);
        let mut read_value = F::zero();
        for i in 0..8 {
            assert_eq!(step.mem_ops[i].t, step.mem_t + i as u64);
            assert_eq!(F::from(step.mem_ops[i].addr), addr + F::from(i as u64));
            assert_eq!(step.mem_ops[i].rw, RW::Read);
            read_value += F::from(step.mem_ops[i].value as u64) * f_pow::<F>(2, 8 * i);
        }
        assert_eq!(step_next.regs[inst.rd], read_value);
        assert_eq!(step_next.pc, step.pc + F::from(4u32));
        assert_eq!(step.mem_ops.len(), 8);
        assert_eq!(step_next.mem_t, step.mem_t + 8);
    }
    // `sd rs2,imm(rs1)`: `memory[rs1 + imm] = rs2; pc = pc + 4` with `-2^11 <= imm < 2^11`
    pub fn t_sd(step: &Step<F>, step_next: &Step<F>) {
        let inst = &step.inst;
        // Ref: Jolt 5.8 (Sum is done as in Branch instructions)
        // Index. z = x + y over the native field
        let z = step.regs[inst.rs1] + F::from(inst.imm);
        // Lookup. z has W+1 bits.  Take lowest W bits via lookup table
        let addr = LookupTables::zero_upper_bits(W + 1, W, W / C, z);
        let mut write_value = F::zero();
        for i in 0..8 {
            assert_eq!(step.mem_ops[i].t, step.mem_t + i as u64);
            assert_eq!(F::from(step.mem_ops[i].addr), addr + F::from(i as u64));
            assert_eq!(step.mem_ops[i].rw, RW::Write);
            write_value += F::from(step.mem_ops[i].value as u64) * f_pow::<F>(2, 8 * i);
        }
        assert_eq!(step.regs[inst.rs2], write_value);
        assert_eq!(step_next.pc, step.pc + F::from(4u32));
        assert_eq!(step.mem_ops.len(), 8);
        assert_eq!(step_next.mem_t, step.mem_t + 8);
    }

    // #### Arithmetic

    // `add rd,rs1,rs2`: `rd = rs1 + rs2; pc = pc + 4`
    pub fn t_add(step: &Step<F>, step_next: &Step<F>) {
        let inst = &step.inst;
        // Ref: Jolt 5.2
        // Index. z = x + y over the native field
        let z = step.regs[inst.rs1] + step.regs[inst.rs2];
        // Lookup. z has W+1 bits.  Take lowest W bits via lookup table
        let result = LookupTables::zero_upper_bits(W + 1, W, W / C, z);
        assert_eq!(step_next.regs[inst.rd], result);
        assert_eq!(step_next.pc, step.pc + F::from(4u32));
        assert_eq!(step.mem_ops.len(), 0);
        assert_eq!(step_next.mem_t, step.mem_t);
    }
    // `sub rd,rs1,rs2`: `rd = rs1 - rs2; pc = pc + 4`
    pub fn t_sub(step: &Step<F>, step_next: &Step<F>) {
        let inst = &step.inst;
        // Ref: Jolt 5.2
        // Index. z = x + (2^W - y) over the native field
        let z = step.regs[inst.rs1] + (f_pow::<F>(2, W) - step.regs[inst.rs2]);
        // Lookup. z has W+1 bits.  Take lowest W bits via lookup table
        let result = LookupTables::zero_upper_bits(W + 1, W, W / C, z);
        assert_eq!(step_next.regs[inst.rd], result);
        assert_eq!(step_next.pc, step.pc + F::from(4u32));
        assert_eq!(step.mem_ops.len(), 0);
        assert_eq!(step_next.mem_t, step.mem_t);
    }
    // `mul rd,rs1,rs2`: `rd = rs1 * rs2; pc = pc + 4`
    pub fn t_mul(step: &Step<F>, step_next: &Step<F>) {
        let inst = &step.inst;
        // Ref: Jolt 6.2.1
        // Index. z = x * y over the native field
        let z = step.regs[inst.rs1] * step.regs[inst.rs2];
        // Lookup. z has 2*W bits.  Take lowest W bits via lookup table
        // let result = LookupTables::wx2_to_w(W, z);
        let result = LookupTables::zero_upper_bits(2 * W, W, W / C, z);
        assert_eq!(step_next.regs[inst.rd], result);
        assert_eq!(step_next.pc, step.pc + F::from(4u32));
        assert_eq!(step.mem_ops.len(), 0);
        assert_eq!(step_next.mem_t, step.mem_t);
    }
    // `divu rd,rs1,rs2`: `rd = rs1 / rs2; pc = pc + 4` where the values of `rs1` and
    // `rs2` are interpreted as unsigned integers.
    pub fn t_divu(step: &Step<F>, step_next: &Step<F>) {
        let inst = &step.inst;
        // Ref: Jolt 6.3
        // x = q * y + r
        // where x = rs1, y = rs2, q = rd
        assert_eq!(step.mem_ops.len(), 0);
        assert_eq!(step_next.mem_t, step.mem_t);
        todo!();
    }
    // `remu rd,rs1,rs2`: `rd = rs1 % rs2; pc = pc + 4` where the values of `rs1` and `rs2` are
    // interpreted as unsigned integers.
    pub fn t_remu(step: &Step<F>, step_next: &Step<F>) {
        let inst = &step.inst;
        // Ref: Jolt 6.3
        // x = q * y + r
        // where x = rs1, y = rs2, r = rd
        assert_eq!(step.mem_ops.len(), 0);
        assert_eq!(step_next.mem_t, step.mem_t);
        todo!();
    }

    // #### Comparison

    // `sltu rd,rs1,rs2`: `if (rs1 < rs2) { rd = 1 } else { rd = 0 } pc = pc + 4`
    // where the values of `rs1` and `rs2` are interpreted as unsigned integers.
    pub fn t_sltu(step: &Step<F>, step_next: &Step<F>) {
        let inst = &step.inst;
        // Ref: Jolt 5.3, 4.2.2
        // Index. z = x || y over the native field
        let z = step.regs[inst.rs1] + f_pow::<F>(2, W) * step.regs[inst.rs2];
        // Lookup
        let result = LookupTables::ltu(W, C, z);
        assert_eq!(step_next.regs[inst.rd], result);
        assert_eq!(step_next.pc, step.pc + F::from(4u32));
        assert_eq!(step.mem_ops.len(), 0);
        assert_eq!(step_next.mem_t, step.mem_t);
    }

    // #### Control

    // `beq rs1,rs2,imm`: `if (rs1 == rs2) { pc = pc + imm } else { pc = pc + 4 }`
    // with `-2^12 <= imm < 2^12` and `imm % 2 == 0`
    pub fn t_beq(step: &Step<F>, step_next: &Step<F>) {
        let inst = &step.inst;
        // Ref: Jolt 5.7
        // Index. z = x || y over the native field
        let z = step.regs[inst.rs1] + f_pow::<F>(2, W) * step.regs[inst.rs2];
        // PC is not computed by a lookup
        let condition = LookupTables::eq(W, C, z);
        // Simulate the encoding of the instruction from Jolt (where opflag=positive)
        let imm = F::from(inst.imm);
        assert_eq!(
            step_next.pc,
            step.pc
                + if condition == F::ONE {
                    if inst.opflags.is_positive {
                        imm
                    } else {
                        -imm
                    }
                } else {
                    F::from(4u32)
                }
        );
        assert_eq!(step.mem_ops.len(), 0);
        assert_eq!(step_next.mem_t, step.mem_t);
    }
    // `jal rd,imm`: `rd = pc + 4; pc = pc + imm` with `-2^20 <= imm < 2^20` and `imm % 2 == 0`
    pub fn t_jal(step: &Step<F>, step_next: &Step<F>) {
        let inst = &step.inst;
        // Ref: Jolt 5.6
        // Index. z = x + y over the native field
        let z = step.pc + F::from(inst.imm);
        // Lookup. z has W+1 bits.  Take lowest W bits via lookup table
        let result = LookupTables::zero_upper_bits(W + 1, W, W / C, z);
        // Errata: Jolt says that the value stored in rd is the new pc + 4, but that wouldn't be
        // useful.  The value stored in rd is the old pc + 4, so that we can return back after the
        // jump.
        assert_eq!(step_next.regs[inst.rd], step.pc + F::from(4u32));
        assert_eq!(step_next.pc, result);
        assert_eq!(step.mem_ops.len(), 0);
        assert_eq!(step_next.mem_t, step.mem_t);
    }
    // `jalr rd,imm(rs1)`: `tmp = ((rs1 + imm) / 2) * 2;
    // rd = pc + 4; pc = tmp` with `-2^11 <= imm < 2^11`
    pub fn t_jalr(step: &Step<F>, step_next: &Step<F>) {
        let inst = &step.inst;
        // Ref: Jolt 5.6
        // Errata: Jolt says it checks z = pc + imm + 4, but it seems wrong because that value is
        // not used for the new pc nor the new rd, so we skip the `+ 4`.
        // Index. z = x + y over the native field
        let z = step.regs[inst.rs1] + F::from(inst.imm);
        // Lookup. z has W+1 bits.  Take lowest W bits via lookup table, setting bit 0 to 0.
        let result = LookupTables::zero_upper_bits_lower_bit(W + 1, W, W / C, z);
        // Errata: Jolt says that the value stored in rd is the new pc + 4, but that wouldn't be
        // useful.  The value stored in rd is the old pc + 4, so that we can return back after the
        // jump.
        assert_eq!(step_next.regs[inst.rd], step.pc + F::from(4u32));
        assert_eq!(step_next.pc, result);
        assert_eq!(step.mem_ops.len(), 0);
        assert_eq!(step_next.mem_t, step.mem_t);
    }
    // #### System

    // `ecall`: system call number is in `a7`, actual parameters are in `a0-a3`, return value is in
    // `a0`.
    pub fn t_ecall(step: &Step<F>, step_next: &Step<F>) {
        println!("DBG: Skipping Simulator ecall");
        assert_eq!(step.mem_ops.len(), 0);
        assert_eq!(step_next.mem_t, step.mem_t);
    }
}
