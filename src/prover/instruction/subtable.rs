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

/// T[X || Y] = T_1[X_1 || Y_1] * T_2[X_2 || Y_2] * 2^8 * ... * T_8[X_8 || Y_8] * 2^56
impl<F: PrimeField> DecomposableTable<F> for EqTable<F> {
    fn num_memories(&self) -> usize {
        8
    }

    // same as SubTableMLE::eq
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
        vec![16; 8]
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

#[derive(Clone, Debug)]
pub struct LtuTable<F>(PhantomData<F>);

impl<F> LtuTable<F> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<F: PrimeField> DecomposableTable<F> for LtuTable<F> {
    // The number of memories is 2c - 1
    fn num_memories(&self) -> usize {
        15
    }

    fn subtable_polys(&self) -> Vec<MultilinearPolynomial<F>> {
        let memory_size = 1 << 16;
        let mut evals_eq = vec![];
        (0..memory_size).for_each(|i| {
            let (lhs, rhs) = split_bits(i, 8);
            let result = F::from((lhs == rhs) as u64);
            evals_eq.push(result)
        });
        let mut evals_ltu = vec![];
        (0..memory_size).for_each(|i| {
            let (lhs, rhs) = split_bits(i, 8);
            let result = F::from((lhs < rhs) as u64);
            evals_ltu.push(result);
        });
        vec![MultilinearPolynomial::new(evals_ltu), MultilinearPolynomial::new(evals_eq)]
    }

    fn chunk_bits(&self) -> Vec<usize> {
        vec![16; 8]
    }

    fn subtable_indices(&self, index_bits: Vec<bool>) -> Vec<Vec<bool>> {
        subtable_indices_bin_op(self.chunk_bits(), index_bits)
    }

    fn combine_lookup_expressions(&self, expressions: Vec<Expression<F>>) -> Expression<F> {
        let c = expressions.len() / 2;
        let ltu = |i| &expressions[i * 2];
        let eq = |i| &expressions[i * 2 + 1];
        let mut result = Expression::Constant(F::ZERO);
        let mut eq_acc = Expression::Constant(F::ONE);

        for i in (0..c).rev() {
            result = result + ltu(i).clone() * eq_acc.clone();
            eq_acc = eq_acc * eq(i).clone();
        }
        result
    }

    fn combine_lookups(&self, operands: &[F]) -> F {
        CombineLookups::ltu(operands)
    }

    fn memory_to_chunk_index(&self, memory_index: usize) -> usize {
        if memory_index % 2 == 1 {
            (memory_index + 1) / 2
        } else {
            memory_index / 2
        }
    }

    fn memory_to_subtable_index(&self, memory_index: usize) -> usize {
        memory_index % 2
    }
}
