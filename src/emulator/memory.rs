use crate::{MemOp, RW};

use elf::abi::PF_X;
use elf::endian::LittleEndian;
use elf::ElfBytes;

pub trait Memory {
    fn read_u8(&mut self, addr: u64) -> u8;

    fn read_u32(&mut self, addr: u64) -> u32 {
        assert!(addr % 4 == 0, "read-address-misaligned");
        let data = [0, 1, 2, 3].map(|i| self.read_u8(addr + i));
        u32::from_le_bytes(data)
    }

    fn read_u64(&mut self, addr: u64) -> u64 {
        assert!(addr % 8 == 0, "read-address-misaligned");
        let data = [0, 1, 2, 3, 4, 5, 6, 7].map(|i| self.read_u8(addr + i));
        u64::from_le_bytes(data)
    }

    fn write_u8(&mut self, addr: u64, value: u8);

    fn write_u64(&mut self, addr: u64, value: u64) {
        assert!(addr % 8 == 0, "write-address-misaligned");
        let data = value.to_le_bytes();
        data.iter()
            .enumerate()
            .for_each(|(i, b)| self.write_u8(addr + i as u64, *b));
    }

    // Return inital stack pointer
    fn sp(&self) -> u64;

    fn entry_point(&self) -> u64;
}

#[derive(Default)]
pub struct NoMem;

impl Memory for NoMem {
    fn read_u8(&mut self, addr: u64) -> u8 {
        panic!("No memory")
    }

    fn write_u8(&mut self, addr: u64, value: u8) {
        panic!("No memory")
    }

    fn sp(&self) -> u64 {
        0
    }

    fn entry_point(&self) -> u64 {
        0
    }
}

// https://github.com/riscv-software-src/riscv-pk
pub struct RiscvPkMemoryMap(Vec<u8>);

pub const RISCV_PK_VMEM: u64 = 0x10000;
pub const RISCV_PK_ENTRY_POINT: u64 = RISCV_PK_VMEM;

impl RiscvPkMemoryMap {
    pub fn new(mem_size: usize) -> Self {
        Self(vec![0; mem_size])
    }

    pub fn new_load(mem_size: usize, text: &[u8], data_addr: usize, data: &[u8]) -> Self {
        let mut mem = RiscvPkMemoryMap::new(mem_size);
        mem.load(text, data_addr, data);
        mem
    }

    pub fn new_load_from_elf(mem_size: usize, file: &ElfBytes<LittleEndian>) -> Self {
        let entry_point = file.ehdr.e_entry;
        assert_eq!(file.ehdr.e_entry, RISCV_PK_ENTRY_POINT);
        let mut text_segment = None;
        let mut data_segment = None;
        let segments = file.segments().expect("Get segments");
        for segment in segments.iter() {
            if segment.p_flags & PF_X != 0 {
                if text_segment.is_some() {
                    panic!("Found 2 executable segments");
                }
                text_segment = Some(segment);
            } else {
                if data_segment.is_some() {
                    panic!("Found 2 data segments");
                }
                data_segment = Some(segment);
            }
        }
        match (text_segment, data_segment) {
            (Some(text_segment), Some(data_segment)) => {
                let text = file.segment_data(&text_segment).expect("Get text bytes");
                let data = file.segment_data(&data_segment).expect("Get data bytes");
                let data_addr = data_segment.p_vaddr;
                Self::new_load(mem_size, text, data_addr as usize, data)
            }
            (None, Some(_)) => panic!("Missing text segment"),
            (Some(_), None) => panic!("Missing data segment"),
            (None, None) => panic!("Missing text & data segment"),
        }
    }

    pub fn load(&mut self, text: &[u8], data_addr: usize, data: &[u8]) {
        assert!(text.len() < self.0.len());
        assert!((data_addr - RISCV_PK_ENTRY_POINT as usize) + data.len() < self.0.len());
        // Load text section
        for (i, b) in text.iter().enumerate() {
            self.write_u8(RISCV_PK_ENTRY_POINT + i as u64, text[i]);
        }
        // Load data section
        for (i, b) in data.iter().enumerate() {
            self.write_u8((data_addr + i) as u64, data[i]);
        }
    }

    // Map address to RAM index
    fn map(&self, addr: u64) -> u64 {
        if addr < RISCV_PK_VMEM {
            panic!("Invalid low address 0x{:x} < 0x{:x}", addr, RISCV_PK_VMEM);
        } else if RISCV_PK_VMEM <= addr && addr < RISCV_PK_VMEM + self.0.len() as u64 {
            addr - RISCV_PK_VMEM
        } else {
            panic!(
                "Invalid high address 0x{:x} > 0x{:x}",
                addr,
                RISCV_PK_VMEM + self.0.len() as u64
            );
        }
    }
}

impl Memory for RiscvPkMemoryMap {
    fn read_u8(&mut self, addr: u64) -> u8 {
        let index = self.map(addr) as usize;
        self.0[index]
    }

    fn write_u8(&mut self, addr: u64, value: u8) {
        let index = self.map(addr) as usize;
        self.0[index] = value;
    }

    fn sp(&self) -> u64 {
        RISCV_PK_ENTRY_POINT + self.0.len() as u64 - std::mem::size_of::<u64>() as u64
    }

    fn entry_point(&self) -> u64 {
        RISCV_PK_ENTRY_POINT
    }
}

#[derive(Debug)]
pub struct MemoryTracer<M: Memory> {
    mem: M,                // Memory state
    pub trace: Vec<MemOp>, // Chronological trace of memory operations
    t: u64,                // Timestamp counter
}

impl<M: Memory> MemoryTracer<M> {
    pub fn new(mem: M) -> Self {
        Self {
            mem,
            trace: Vec::new(),
            t: 0,
        }
    }
}

impl<M: Memory> Memory for MemoryTracer<M> {
    fn read_u8(&mut self, addr: u64) -> u8 {
        let value = self.mem.read_u8(addr);
        self.trace.push(MemOp {
            addr,
            t: self.t,
            rw: RW::Read,
            value,
        });
        self.t += 1;
        value
    }

    fn write_u8(&mut self, addr: u64, value: u8) {
        self.trace.push(MemOp {
            addr,
            t: self.t,
            rw: RW::Write,
            value,
        });
        self.t += 1;
        self.mem.write_u8(addr, value)
    }

    fn sp(&self) -> u64 {
        self.mem.sp()
    }

    fn entry_point(&self) -> u64 {
        self.mem.entry_point()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_new_load_from_elf() {
        let path = std::path::PathBuf::from("riscu_examples/c/fibo.bin");
        let file_data = std::fs::read(path).expect("Could not read file.");
        let slice = file_data.as_slice();
        let file = ElfBytes::<LittleEndian>::minimal_parse(slice).expect("Open test1");
        let mem = RiscvPkMemoryMap::new_load_from_elf(1024 * 1024, &file);
    }
}
