use crate::emulator::Emulator as GenericEmulator;
use crate::simulator::Simulator;

use ark_bn254::fr::Fr;
use ark_ff::{Field, PrimeField};

type Emulator = GenericEmulator<u64>;

fn extend_cases_commutative(cases: &mut Vec<(u64, u64, u64)>) {
    let commuted: Vec<(u64, u64, u64)> = cases
        .iter()
        .cloned()
        .map(|(a, b, result)| (b, a, result))
        .collect();
    cases.extend_from_slice(&commuted);
}

// Test emulator and simulator behaviour against a vector of cases consisting of (result, a, b)
// where `result = a op b`.  `emu_inst` and `sim_inst` are function pointers to the emulador and
// simulator implementations of the instruction.
fn test_emu_vs_sim<F: PrimeField>(
    inst_str: &str,
    emu_inst: fn(&mut Emulator, usize, usize, usize),
    sim_inst: fn(&mut Simulator<F>, usize, usize, usize),
    cases: &Vec<(u64, u64, u64)>,
) {
    for (result, a, b) in cases.iter().cloned() {
        let mut emu = Emulator::default();
        let mut sim = Simulator::default();

        emu.regs[2] = a;
        emu.regs[3] = b;
        emu_inst(&mut emu, 1, 2, 3);
        let r = emu.regs[1];
        assert_eq!(r, result, "emu {a} {inst_str} {b} = {result} != {r}");

        sim.regs[2] = F::from(a);
        sim.regs[3] = F::from(b);
        sim_inst(&mut sim, 1, 2, 3);
        let r = sim.regs[1];
        assert_eq!(
            F::from(result),
            r,
            "sim {a} {inst_str} {b} = {result} != {r}"
        );
    }
}

#[test]
fn test_add() {
    // result = a + b
    let cases: Vec<(u64, u64, u64)> = [
        (0, 0, 0),
        (5, 0, 5),
        (8, 8, 0),
        (10, 3, 7),
        (0xffffffffffffffff, 0xffffffffffffffff, 0),
        (0xffffffffffffffff, 0xfffffffffffffffe, 1),
        (0, 0xffffffffffffffff, 1),
        (1, 0xffffffffffffffff, 2),
        (0xfffffffffffffffe, 0xffffffffffffffff, 0xffffffffffffffff),
    ]
    .to_vec();

    test_emu_vs_sim::<Fr>(&"add", Emulator::add, Simulator::t_add, &cases);
}

#[test]
fn test_sub() {
    // result = a - b
    let cases: Vec<(u64, u64, u64)> = [
        (0, 0, 0),
        (5, 5, 0),
        (0, 8, 8),
        (7, 10, 3),
        (0, 0xffffffffffffffff, 0xffffffffffffffff),
        (0xfffffffffffffffe, 0xffffffffffffffff, 1),
        (0xffffffffffffffff, 0, 1),
        (0xffffffffffffffff, 1, 2),
        (0xffffffffffffffff, 0xfffffffffffffffe, 0xffffffffffffffff),
    ]
    .to_vec();

    test_emu_vs_sim::<Fr>(&"sub", Emulator::sub, Simulator::t_sub, &cases);
}

#[test]
fn test_mul() {
    // result = a * b
    let cases: Vec<(u64, u64, u64)> = [
        (0, 0, 0),
        (0, 5, 0),
        (63, 7, 9),
        (30, 10, 3),
        (73, 73, 1),
        (0, 0x100000000, 0x100000000),
        (0x1200000012, 0x100000001, 0x12),
        (0xfffffffffffffffe, 0xffffffffffffffff, 2),
        (0x1, 0xffffffffffffffff, 0xffffffffffffffff),
    ]
    .to_vec();

    test_emu_vs_sim::<Fr>(&"mul", Emulator::mul, Simulator::t_mul, &cases);
}

#[test]
fn test_divu() {
    // result = a / b
    let cases: Vec<(u64, u64, u64)> = [
        (0xffffffffffffffff, 0, 0),
        (0xffffffffffffffff, 5678, 0),
        (1, 1, 1),
        (2, 2, 1),
        (5, 10, 2),
        (1387, 1234567, 890),
        (1, 0xffffffffffffffff, 0xffffffffffffffff),
        (0x7fffffffffffffff, 0xffffffffffffffff, 2),
    ]
    .to_vec();

    test_emu_vs_sim::<Fr>(&"divu", Emulator::divu, Simulator::t_divu, &cases);
}

#[test]
fn test_remu() {
    // result = a % b
    let cases: Vec<(u64, u64, u64)> = [
        (0, 0, 0),
        (56, 56, 0),
        (1, 11, 10),
        (444, 444, 445),
        (0, 444, 444),
        (56, 123456, 1234),
        (0, 0xffffffffffffffff, 0xffffffffffffffff),
        (0xfffffffffffffffe, 0xfffffffffffffffe, 0xffffffffffffffff),
        (1, 0xffffffffffffffff, 0xfffffffffffffffe),
    ]
    .to_vec();

    test_emu_vs_sim::<Fr>(&"remu", Emulator::remu, Simulator::t_remu, &cases);
}

#[test]
fn test_sltu() {
    // result = a < b
    let cases: Vec<(u64, u64, u64)> = [
        (0, 0, 0),
        (1, 0, 500),
        (0, 400, 0),
        (0, 999, 999),
        (0, 0xffffffffffffffff, 0xffffffffffffffff),
        (1, 0xfffffffffffffffe, 0xffffffffffffffff),
        (0, 0xffffffffffffffff, 0xfffffffffffffffe),
        (1, 0x000000000000fffe, 0x000000000000ffff),
        (0, 0x000000000001fffe, 0x000000000000ffff),
    ]
    .to_vec();

    test_emu_vs_sim::<Fr>(&"sltu", Emulator::sltu, Simulator::t_sltu, &cases);
}

use crate::expr::{Expr, Var};
use crate::simulator::SubTableMLE;
use ark_ff::{biginteger::BigInteger, One, Zero};
use std::fmt::{self, Display};

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Bit {
    var: usize,
    index: usize,
}

impl Var for Bit {}

const VARS: &str = "xyzabcdefghijklmnopqrstuvw";

impl Display for Bit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        assert!(self.var < VARS.len());
        write!(f, "{}_{}", &VARS[self.var..self.var + 1], self.index)
    }
}

#[test]
fn test_expr() {
    let w: usize = 64;
    let c: usize = 16;
    let chunk_len = w / c;

    let x: Vec<Expr<Fr, Bit>> = (0..chunk_len)
        .map(|i| Expr::Var(Bit { var: 0, index: i }))
        .collect();
    let y: Vec<Expr<Fr, Bit>> = (0..chunk_len)
        .map(|i| Expr::Var(Bit { var: 1, index: i }))
        .collect();
    let mut eq = SubTableMLE::eq_mle(&x, &y);
    println!("{}", eq);
    let terms = eq.normalize();
    println!("{}", terms);
}
