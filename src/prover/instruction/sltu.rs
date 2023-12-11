use halo2_curves::ff::PrimeField;

use plonkish_backend::{
    backend::lookup::lasso::DecomposableTable, poly::multilinear::MultilinearPolynomial,
    util::expression::Expression,
};

use super::subtable::LtuTable;

#[derive(Clone, Debug)]
pub struct SltuTable<F> {
    ltu: LtuTable<F>,
}

impl<F> SltuTable<F> {
    pub fn new() -> Self {
        Self { ltu: LtuTable::new() }
    }
}

/// T[X || Y] = T_1[X_1 || Y_1] + T_2[X_2 || Y_2] * 2^11 + ... + T_8[X_8 || Y_8] * 2^56
impl<F: PrimeField> DecomposableTable<F> for SltuTable<F> {
    fn num_memories(&self) -> usize {
        8
    }

    fn subtable_polys(&self) -> Vec<MultilinearPolynomial<F>> {
        self.ltu.subtable_polys()
    }

    fn chunk_bits(&self) -> Vec<usize> {
        self.ltu.chunk_bits()
    }

    fn subtable_indices(&self, index_bits: Vec<bool>) -> Vec<Vec<bool>> {
        self.ltu.subtable_indices(index_bits)
    }

    fn combine_lookup_expressions(&self, expressions: Vec<Expression<F>>) -> Expression<F> {
        self.ltu.combine_lookup_expressions(expressions)
    }

    fn combine_lookups(&self, operands: &[F]) -> F {
        self.ltu.combine_lookups(operands)
    }

    fn memory_to_chunk_index(&self, memory_index: usize) -> usize {
        self.ltu.memory_to_chunk_index(memory_index)
    }

    fn memory_to_subtable_index(&self, memory_index: usize) -> usize {
        self.ltu.memory_to_chunk_index(memory_index)
    }
}
