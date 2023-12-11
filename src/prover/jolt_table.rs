use halo2_curves::ff::PrimeField;

use plonkish_backend::{
    backend::lookup::lasso::DecomposableTable, poly::multilinear::MultilinearPolynomial,
    util::{expression::Expression, arithmetic::usize_from_bits_le},
};
use crate::Opcode;

const OPCODE_BIT: usize = 4;

fn get_opcode(index_bits: &Vec<bool>) -> Opcode {
    let opcode = usize_from_bits_le(&index_bits[..OPCODE_BIT - 1]);
    Opcode::from_usize(opcode)
}

fn get_operand(index_bits: Vec<bool>) -> Vec<bool> {
    index_bits[OPCODE_BIT..].to_vec()
}

#[derive(Clone, Debug)]
pub struct JoltTable<F>{
    tables: Vec<Box<dyn DecomposableTable<F>>>
}

impl<F: PrimeField> JoltTable<F> {
    pub fn compose(tables: Vec<Box<dyn DecomposableTable<F>>>) -> Self {
        let first_chunk_bits = tables[0].chunk_bits();
        assert!(tables.iter().all(|t| t.chunk_bits() == first_chunk_bits));
        Self { tables }
    }
}

/// `JoltTable` is composition of all instruction evaluation tables
impl<F: PrimeField> DecomposableTable<F> for JoltTable<F> {
    /// Take maximum number of lookups for subtable among instructions
    fn num_memories(&self) -> usize {
        self.tables.iter().map(|t| t.num_memories()).max().unwrap()
    }

    fn subtable_polys(&self) -> Vec<MultilinearPolynomial<F>> {
        let mut subtables = self.tables[0].subtable_polys();
        for s in self.tables[1..].iter() {
            subtables.append(&mut s.subtable_polys())
        }
        subtables
    }

    fn chunk_bits(&self) -> Vec<usize> {
        vec![16; 8]
    }

    fn subtable_indices(&self, index_bits: Vec<bool>) -> Vec<Vec<bool>> {
        let opcode: usize = get_opcode(&index_bits).into();
        let operand = get_operand(index_bits);
        self.tables[opcode].subtable_indices(operand)
    }

    fn combine_lookup_expressions(&self, expressions: Vec<Expression<F>>) -> Expression<F> {
        Expression::DistributePowers(expressions, Box::new(Expression::Constant(F::from(1 << 8))))
    }

    /// g(w, x) = Σ(EQ(w, x)*g_instr(x))
    /// in this case, there are 14 opcodes
    /// `g` takes 4 bits long w and 128 bits x
    fn combine_lookups(&self, operands: &[F]) -> F {
        F::ZERO
    }

    fn memory_to_chunk_index(&self, memory_index: usize) -> usize {
        memory_index
    }

    fn memory_to_subtable_index(&self, memory_index: usize) -> usize {
        0
    }
}

#[cfg(test)]
mod test {
    #[test]
    #[ignore]
    fn test_composition() {

    }
}
