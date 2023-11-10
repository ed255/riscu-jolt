use ff::PrimeField;
use num_bigint::BigUint;

pub fn into_bigint<F: PrimeField>(i: &F) -> BigUint {
    BigUint::from_bytes_le(i.to_repr().as_ref())
}

pub fn num_bits<F: PrimeField>(i: &F) -> u64 {
    into_bigint(i).bits()
}
