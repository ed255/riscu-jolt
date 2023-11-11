use std::marker::PhantomData;

use halo2_curves::ff::PrimeField;

use plonkish_backend::{
    backend::lookup::lasso::DecomposableTable, poly::multilinear::MultilinearPolynomial,
    util::expression::Expression,
};

use crate::simulator::CombineLookups;

use super::subtable_indices_identity;

#[derive(Clone, Debug)]
pub struct JalrTable<F>(PhantomData<F>);

impl<F> JalrTable<F> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}
/// `JALR`'s evaluation result of pc is from its own calculation on finite field
/// this table does range check
/// T[z] = T_1[z_1] + T_2[z_2] * 2^8 + ... + T_8[z_8] * 2^56
impl<F: PrimeField> DecomposableTable<F> for JalrTable<F> {
    fn num_memories(&self) -> usize {
        8
    }

    fn subtable_polys(&self) -> Vec<MultilinearPolynomial<F>> {
        let memory_size_identity = 1 << 16;
        let mut evals_identity = vec![];
        (0..memory_size_identity).for_each(|i| {
            let result = F::from(i as u64);
            evals_identity.push(result)
        });
        vec![MultilinearPolynomial::new(evals_identity)]
    }

    fn chunk_bits(&self) -> Vec<usize> {
        vec![16; 8]
    }

    fn subtable_indices(&self, index_bits: Vec<bool>) -> Vec<Vec<bool>> {
        subtable_indices_identity(self.chunk_bits(), index_bits)
    }

    fn combine_lookup_expressions(&self, expressions: Vec<Expression<F>>) -> Expression<F> {
        Expression::DistributePowers(expressions, Box::new(Expression::Constant(F::from(1 << 8))))
    }

    fn combine_lookups(&self, operands: &[F]) -> F {
        CombineLookups::range(16, operands)
    }

    fn memory_to_chunk_index(&self, memory_index: usize) -> usize {
        memory_index
    }

    fn memory_to_subtable_index(&self, memory_index: usize) -> usize {
        0
    }
}
