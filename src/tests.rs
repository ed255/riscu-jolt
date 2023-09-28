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

#[test]
fn test_add() {
    // a + b = result
    let cases: Vec<(u64, u64, u64)> = [
        // (0, 0, 0),
        (0, 5, 5),
        (8, 0, 8),
        (3, 7, 10),
        (0xffffffffffffffff, 0, 0xffffffffffffffff),
        (0xfffffffffffffffe, 1, 0xffffffffffffffff),
        (0xffffffffffffffff, 1, 0),
        (0xffffffffffffffff, 2, 1),
        (0xffffffffffffffff, 0xffffffffffffffff, 0xfffffffffffffffe),
    ]
    .to_vec();

    for (a, b, result) in cases.iter().cloned() {
        let mut emu = Emulator::default();
        let mut sim = Simulator::default();

        emu.regs[2] = a;
        emu.regs[3] = b;
        emu.add(1, 2, 3);
        let r = emu.regs[1];
        assert_eq!(r, result, "{a} + {b} = {result} != {r}");

        sim.regs[2] = F::from(a);
        sim.regs[3] = F::from(b);
        sim.t_add(1, 2, 3);
        let r = sim.regs[1];
        assert_eq!(r, F::from(result), "{a} + {b} = {result} != {r}");
    }
}
