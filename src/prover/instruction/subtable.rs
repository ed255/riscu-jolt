use std::marker::PhantomData;

use halo2_curves::ff::PrimeField;

use plonkish_backend::{
    backend::lookup::lasso::DecomposableTable,
    poly::multilinear::MultilinearPolynomial,
    util::{arithmetic::split_bits, expression::Expression},
};

use super::subtable_indices_bin_op;
use crate::simulator::CombineLookups;

#[derive(Clone, Debug)]
pub struct EqTable<F>(PhantomData<F>);

impl<F> EqTable<F> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

/// T[X || Y] = T_1[X_1 || Y_1] + T_2[X_2 || Y_2] * 2^8 + ... + T_8[X_8 || Y_8] * 2^56
impl<F: PrimeField> DecomposableTable<F> for EqTable<F> {
    fn num_memories(&self) -> usize {
        6
    }

    fn subtable_polys(&self) -> Vec<MultilinearPolynomial<F>> {
        let memory_size = 1 << 16;
        let mut evals = vec![];
        (0..memory_size).for_each(|i| {
            let (lhs, rhs) = split_bits(i, 8);
            let result = F::from((lhs == rhs) as u64);
            evals.push(result)
        });
        vec![MultilinearPolynomial::new(evals)]
    }

    fn chunk_bits(&self) -> Vec<usize> {
        vec![20, 20, 22, 22, 22, 22]
    }

    fn subtable_indices(&self, index_bits: Vec<bool>) -> Vec<Vec<bool>> {
        subtable_indices_bin_op(self.chunk_bits(), index_bits)
    }

    fn combine_lookup_expressions(&self, expressions: Vec<Expression<F>>) -> Expression<F> {
        expressions.iter().product()
    }

    fn combine_lookups(&self, operands: &[F]) -> F {
        CombineLookups::eq(operands)
    }

    fn memory_to_chunk_index(&self, memory_index: usize) -> usize {
        memory_index
    }

    fn memory_to_subtable_index(&self, memory_index: usize) -> usize {
        0
    }
}
