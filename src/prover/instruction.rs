pub mod add;
pub mod and;
pub mod beq;
pub mod jal;
pub mod jalr;
pub mod mul;
pub mod sltu;
pub mod sub;

pub mod subtable;

use ff::PrimeField;
use itertools::{izip, Itertools};
use plonkish_backend::util::arithmetic::{inner_product, split_by_chunk_bits};
use std::iter;

fn subtable_indices_bin_op(chunk_bits: Vec<usize>, index_bits: Vec<bool>) -> Vec<Vec<bool>> {
    assert!(index_bits.len() % 2 == 0);
    let chunk_bits = chunk_bits
        .iter()
        .map(|chunk_bits| chunk_bits / 2)
        .collect_vec();
    let (lhs, rhs) = index_bits.split_at(index_bits.len() / 2);
    izip!(
        split_by_chunk_bits(lhs, &chunk_bits),
        split_by_chunk_bits(rhs, &chunk_bits)
    )
    .map(|(chunked_lhs_bits, chunked_rhs_bits)| {
        iter::empty()
            .chain(chunked_lhs_bits)
            .chain(chunked_rhs_bits)
            .collect_vec()
    })
    .collect_vec()
}

fn subtable_indices_identity(chunk_bits: Vec<usize>, index_bits: Vec<bool>) -> Vec<Vec<bool>> {
    split_by_chunk_bits(index_bits.as_slice(), &chunk_bits)
}

fn combine_lookups_bin_op<F: PrimeField>(operands: &[F]) -> F {
    let weight = F::from(1 << 8);
    inner_product(
        operands,
        iter::successors(Some(F::ONE), |power_of_weight| {
            Some(*power_of_weight * weight)
        })
        .take(operands.len())
        .collect_vec()
        .iter(),
    )
}
