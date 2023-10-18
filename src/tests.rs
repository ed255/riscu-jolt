use crate::emulator::{memory::NoMem, Emulator as GenericEmulator};
use crate::simulator::Simulator;

use ark_bn254::fr::Fr;
use ark_ff::PrimeField;

type Emulator = GenericEmulator<u64, NoMem>;

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
        assert_eq!(result, r, "emu {a} {inst_str} {b} = {result} != {r}");

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
fn test_inst_add() {
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
fn test_inst_sub() {
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
fn test_inst_mul() {
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
fn test_inst_divu() {
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
fn test_inst_remu() {
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
fn test_inst_sltu() {
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

#[test]
fn test_inst_addi() {
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

    let inst_str = "addi";
    // rd, rs1, imm
    for (result, a, b) in cases.iter().cloned() {
        let b = b as i64;
        let mut emu = Emulator::default();
        let mut sim = Simulator::default();

        emu.regs[2] = a;
        Emulator::addi(&mut emu, 1, 2, b);
        let r = emu.regs[1];
        assert_eq!(r, result, "emu {a} {inst_str} {b} = {result} != {r}");

        sim.regs[2] = Fr::from(a);
        Simulator::t_addi(&mut sim, 1, 2, b);
        let r = sim.regs[1];
        assert_eq!(
            r,
            Fr::from(result),
            "sim {a} {inst_str} {b} = {result} != {r}"
        );
    }
}

#[test]
fn test_inst_lui() {
    // result = a
    let cases: Vec<u64> = [0, 1 << 12, 3 << 12, 0b11111111111111111111000000000000].to_vec();

    let inst_str = "lui";
    for v in cases.iter().cloned() {
        let v = v as i64;
        let mut emu = Emulator::default();
        let mut sim = Simulator::default();

        Emulator::lui(&mut emu, 1, v);
        let r = emu.regs[1];
        assert_eq!(v as u64, r, "emu {inst_str} {v} = {v} != {r}");

        Simulator::t_lui(&mut sim, 1, v);
        let r = sim.regs[1];
        assert_eq!(Fr::from(v as u64), r, "sim {inst_str} {v} = {v} != {r}");
    }
}

#[test]
fn test_inst_beq() {
    // pc, old_pc, imm, a == b
    let cases: Vec<(u64, u64, i64, u64, u64)> = [
        (0x104, 0x100, 0x18, 0, 1),
        (0x104, 0x100, 0x18, 1234, 1233),
        (0x104, 0x100, 0x18, 0xffffffffffffffff, 0xfffffffffffffffe),
        (0x104, 0x100, -0x18, 0, 1),
        (0x104, 0x100, -0x90, 1234, 1233),
        (0x104, 0x100, -0x04, 0xffffffffffffffff, 0xfffffffffffffffe),
        (0x118, 0x100, 0x18, 0, 0),
        (0x118, 0x100, 0x18, 1234, 1234),
        (0x118, 0x100, 0x18, 0xfffffffffffffffe, 0xfffffffffffffffe),
        (0xe8, 0x100, -0x18, 0, 0),
        (0x70, 0x100, -0x90, 1234, 1234),
        (0xfc, 0x100, -0x04, 0xfffffffffffffffe, 0xfffffffffffffffe),
    ]
    .to_vec();

    let inst_str = "beq";
    for (pc, old_pc, imm, a, b) in cases.iter().cloned() {
        let mut emu = Emulator::default();
        let mut sim = Simulator::default();

        emu.pc = old_pc;
        emu.regs[1] = a;
        emu.regs[2] = b;
        Emulator::beq(&mut emu, 1, 2, imm);
        let r = emu.pc;
        assert_eq!(pc, r, "emu {a} {inst_str} {b} -> pc={pc} != pc={r}");

        sim.pc = Fr::from(old_pc);
        sim.regs[1] = Fr::from(a);
        sim.regs[2] = Fr::from(b);
        Simulator::t_beq(&mut sim, 1, 2, imm);
        let r = sim.pc;
        assert_eq!(
            Fr::from(pc),
            r,
            "sim {a} {inst_str} {b} -> pc={pc} != pc={r}"
        );
    }
}

#[test]
fn test_inst_jal() {
    // pc, old_pc, imm
    let cases: Vec<(u64, u64, i64)> = [
        (0x118, 0x100, 0x18),
        (0x218, 0x200, 0x18),
        (0x1018, 0x1000, 0x18),
        (0xe8, 0x100, -0x18),
        (0x70, 0x100, -0x90),
        (0xfc, 0x100, -0x04),
    ]
    .to_vec();

    let inst_str = "jal";
    for (pc, old_pc, imm) in cases.iter().cloned() {
        let mut emu = Emulator::default();
        let mut sim = Simulator::default();

        emu.pc = old_pc;
        Emulator::jal(&mut emu, 1, imm);
        let r = emu.pc;
        let rd = emu.regs[1];
        assert_eq!(pc, r, "emu {inst_str} {imm} -> pc={pc} != pc={r}");
        assert_eq!(old_pc + 4, rd);

        sim.pc = Fr::from(old_pc);
        Simulator::t_jal(&mut sim, 1, imm);
        let r = sim.pc;
        let rd = sim.regs[1];
        assert_eq!(Fr::from(pc), r, "sim {inst_str} {imm} -> pc={pc} != pc={r}");
        assert_eq!(Fr::from(old_pc + 4), rd);
    }
}

#[test]
fn test_inst_jalr() {
    // pc, rs1, imm
    let cases: Vec<(u64, u64, i64)> = [
        (0x118, 0x100, 0x18),
        (0x118, 0x100, 0x19),
        (0x218, 0x200, 0x18),
        (0x218, 0x200, 0x19),
        (0x1018, 0x1000, 0x18),
        (0x1018, 0x1000, 0x19),
        (0xe8, 0x100, -0x18),
        (0xe8, 0x100, -0x17),
        (0x70, 0x100, -0x90),
        (0x70, 0x100, -0x8f),
        (0xfc, 0x100, -0x04),
        (0xfc, 0x100, -0x03),
    ]
    .to_vec();

    let inst_str = "jalr";
    for (pc, rs1, imm) in cases.iter().cloned() {
        let old_pc = 0x100;
        let mut emu = Emulator::default();
        let mut sim = Simulator::default();

        emu.pc = old_pc;
        emu.regs[2] = rs1;
        Emulator::jalr(&mut emu, 1, 2, imm);
        let r = emu.pc;
        let rd = emu.regs[1];
        assert_eq!(pc, r, "emu {inst_str} {imm} -> pc={pc} != pc={r}");
        assert_eq!(old_pc + 4, rd);

        sim.pc = Fr::from(old_pc);
        sim.regs[2] = Fr::from(rs1);
        Simulator::t_jalr(&mut sim, 1, 2, imm);
        let r = sim.pc;
        let rd = sim.regs[1];
        assert_eq!(Fr::from(pc), r, "sim {inst_str} {imm} -> pc={pc} != pc={r}");
        assert_eq!(Fr::from(old_pc + 4), rd);
    }
}

use crate::expr::Expr;
use crate::simulator::{Bit, CombineLookups, SubTableMLE, TableEval};

// Print the MLE for EQ
#[test]
fn test_mle_eq_expr() {
    let w: usize = 64;
    let c: usize = 16;
    let chunk_len = w / c;

    let x: Vec<Expr<Fr, Bit>> = (0..chunk_len)
        .map(|i| Expr::Var(Bit { var: 0, index: i }))
        .collect();
    let y: Vec<Expr<Fr, Bit>> = (0..chunk_len)
        .map(|i| Expr::Var(Bit { var: 1, index: i }))
        .collect();
    let eq = SubTableMLE::eq(&x, &y);
    println!("{}", eq);
    let terms = eq.normalize();
    println!("{}", terms);
}

// Print the `g` expression for LTU
#[test]
fn test_g_ltu_expr() {
    let w: usize = 64;
    let c: usize = 16;
    let chunk_len = w / c;

    let evals_ltu: Vec<Expr<Fr, TableEval>> = (0..chunk_len)
        .map(|i| {
            Expr::Var(TableEval {
                table: "LT",
                chunk: i,
            })
        })
        .collect();
    let evals_eq: Vec<Expr<Fr, TableEval>> = (0..chunk_len)
        .map(|i| {
            Expr::Var(TableEval {
                table: "EQ",
                chunk: i,
            })
        })
        .collect();
    let evals: Vec<Expr<Fr, TableEval>> = evals_ltu
        .into_iter()
        .zip(evals_eq.into_iter())
        .flat_map(|(a, b)| [a, b])
        .collect();
    let ltu_g = CombineLookups::ltu(&evals);
    println!("{}", ltu_g);
    let terms = ltu_g.normalize();
    println!("{}", terms);
}
