#![allow(clippy::identity_op)]

use riscu::emulator::decoder::decode;
use riscu::emulator::{memory::Memory, memory::RiscvPkMemoryMap, Emulator};
use riscu::simulator::{JoltInstruction, Simulator};
use riscu::{Instruction, Opcode, Step as GenericStep};

use halo2_curves::bn256::Fr;

type JoltStep<F> = GenericStep<F, JoltInstruction>;
type EmuStep = GenericStep<u64, Instruction>;

use elf::endian::LittleEndian;
use elf::ElfBytes;

const HELP: &str = "\
emulator

USAGE:
  emulator [OPTIONS] [ELF_PATH]

OPTIONS:
  --debug NUMBER       Debug level
  --sim                Enable verification of the simulator
  --virt               Enable Jolt virtual extension

ARGS:
  <ELF_PATH>
";

#[derive(Debug)]
struct AppArgs {
    simulate: bool,
    virt: bool,
    debug: usize,
    elf_path: std::path::PathBuf,
}

fn main() -> Result<(), pico_args::Error> {
    let mut pargs = pico_args::Arguments::from_env();
    if pargs.contains(["-h", "--help"]) {
        print!("{}", HELP);
        std::process::exit(0);
    }
    let args = AppArgs {
        simulate: pargs.contains("--sim"),
        virt: pargs.contains("--virt"),
        debug: pargs.opt_value_from_str("--debug")?.unwrap_or(0),
        elf_path: pargs.free_from_str()?,
    };
    let remaining = pargs.finish();
    if !remaining.is_empty() {
        eprintln!("Warning: unused arguments left: {:?}.", remaining);
    }

    let file_data = std::fs::read(args.elf_path).expect("Could not read file.");
    let slice = file_data.as_slice();
    let file = ElfBytes::<LittleEndian>::minimal_parse(slice).expect("Open test1");
    // 1 MiB memory
    let mem = RiscvPkMemoryMap::new_load_from_elf(1 * 1024 * 1024, &file);
    let mut emu = Emulator::<u64, _>::new_tracer(mem, args.virt);
    let mut t: u64 = 0;
    loop {
        if args.debug >= 1 {
            let inst_u32 = emu.mem.read_u32(emu.pc);
            // Decode
            let inst = decode(inst_u32);
            println!("t={:05} 0x{:06x} {:08x} {:?}", t, emu.pc, inst_u32, inst);
        }
        let step = emu.step_trace();
        if args.debug >= 2 {
            println!("{:?}", step);
        }
        if args.simulate {
            let step_next = EmuStep {
                pc: emu.pc,
                vpc: emu.virt.as_ref().map(|v| v.pc).unwrap_or(0),
                inst: Instruction::default(),
                regs: emu.regs.clone(),
                mem_ops: Vec::new(),
                mem_t: emu.mem.t,
                advice: 0,
            };
            let step: JoltStep<Fr> = step.into();
            let step_next: JoltStep<Fr> = step_next.into();
            // TODO: Replace this by the step simulation once it's implemented:
            // https://github.com/ed255/riscu-jolt/issues/6
            use Opcode::*;
            match step.inst.op {
                Lui => Simulator::t_lui(&step, &step_next),
                Addi => Simulator::t_addi(&step, &step_next),
                Ld => Simulator::t_ld(&step, &step_next),
                Sd => Simulator::t_sd(&step, &step_next),
                Add => Simulator::t_add(&step, &step_next),
                Sub => Simulator::t_sub(&step, &step_next),
                Mul => Simulator::t_mul(&step, &step_next),
                Divu => Simulator::t_divu(&step, &step_next),
                Remu => Simulator::t_remu(&step, &step_next),
                Sltu => Simulator::t_sltu(&step, &step_next),
                Beq => Simulator::t_beq(&step, &step_next),
                Jal => Simulator::t_jal(&step, &step_next),
                Jalr => Simulator::t_jalr(&step, &step_next),
                Ecall => Simulator::t_ecall(&step, &step_next),
                Virtual(_vop) => todo!(),
            }
        }
        t += 1;
    }
}
