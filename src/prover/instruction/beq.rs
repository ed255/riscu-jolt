use halo2_curves::ff::PrimeField;

use plonkish_backend::{
    backend::lookup::lasso::DecomposableTable, poly::multilinear::MultilinearPolynomial,
    util::expression::Expression,
};

use super::subtable::EqTable;

#[derive(Clone, Debug)]
pub struct BeqTable<F> {
    eq: EqTable<F>,
}

impl<F> BeqTable<F> {
    pub fn new() -> Self {
        Self { eq: EqTable::new() }
    }
}

/// T[X || Y] = T_1[X_1 || Y_1] + T_2[X_2 || Y_2] * 2^11 + ... + T_6[X_6 || Y_6] * 2^58
/// check equality
impl<F: PrimeField> DecomposableTable<F> for BeqTable<F> {
    fn num_memories(&self) -> usize {
        8
    }

    fn subtable_polys(&self) -> Vec<MultilinearPolynomial<F>> {
        self.eq.subtable_polys()
    }

    fn chunk_bits(&self) -> Vec<usize> {
        self.eq.chunk_bits()
    }

    fn subtable_indices(&self, index_bits: Vec<bool>) -> Vec<Vec<bool>> {
        self.eq.subtable_indices(index_bits)
    }

    fn combine_lookup_expressions(&self, expressions: Vec<Expression<F>>) -> Expression<F> {
        self.eq.combine_lookup_expressions(expressions)
    }

    fn combine_lookups(&self, operands: &[F]) -> F {
        self.eq.combine_lookups(operands)
    }

    fn memory_to_chunk_index(&self, memory_index: usize) -> usize {
        memory_index
    }

    fn memory_to_subtable_index(&self, memory_index: usize) -> usize {
        if memory_index == 0 || memory_index == 1 {
            0
        } else {
            1
        }
    }
}
