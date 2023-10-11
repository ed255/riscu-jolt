use riscu::emulator::decoder::decode;
use riscu::emulator::{Emulator, Memory};

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
    let mut emu = Emulator::<u64, _>::new_pk_from_elf(1024 * 1024, &file);
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
            println!("0x{:06x} {:08x} {:?}", emu.pc, inst_u32, inst);
        }
        let pc = emu.pc;
        emu.step();
        if emu.pc == pc {
            // Infinite loop is treated as halt
            break;
        }
    }
    Ok(())
}
