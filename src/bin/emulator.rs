use riscu::emulator::decoder::decode;
use riscu::emulator::{
    memory::Memory,
    memory::{MemoryTracer, RiscvPkMemoryMap},
    Emulator,
};

use elf::endian::LittleEndian;
use elf::ElfBytes;

const HELP: &str = "\
emulator

USAGE:
  emulator [OPTIONS] [ELF_PATH]

OPTIONS:
  --debug NUMBER       Debug level

ARGS:
  <ELF_PATH>
";

#[derive(Debug)]
struct AppArgs {
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
    let mem = MemoryTracer::new(mem);
    let mut emu = Emulator::<u64, _>::new(mem);
    let mut t: u64 = 0;
    loop {
        if args.debug >= 2 {
            for i in 0..10 {
                print!("{:06x} ", emu.regs[i]);
            }
            println!("");
            for i in 10..20 {
                print!("{:06x} ", emu.regs[i]);
            }
            println!("");
        }
        if args.debug >= 1 {
            let inst_u32 = emu.mem.read_u32(emu.pc);
            // Decode
            let inst = decode(inst_u32);
            println!("t={:05} 0x{:06x} {:08x} {:?}", t, emu.pc, inst_u32, inst);
        }
        if t == 00238 {
            for mem_op in &emu.mem.trace {
                println!("{:?}", mem_op);
            }
        }
        emu.step();
        t += 1;
    }
}
