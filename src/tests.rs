use crate::emulator::Emulator;
use crate::simulator::Simulator;

use ark_bn254::fr::Fr as F;

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
fn test_emu_vs_sim(
    inst_str: &str,
    emu_inst: fn(&mut Emulator, usize, usize, usize),
    sim_inst: fn(&mut Simulator, usize, usize, usize),
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
            r,
            F::from(result),
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

    test_emu_vs_sim(&"add", Emulator::add, Simulator::t_add, &cases);
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

    test_emu_vs_sim(&"sub", Emulator::sub, Simulator::t_sub, &cases);
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

    test_emu_vs_sim(&"mul", Emulator::mul, Simulator::t_mul, &cases);
}
