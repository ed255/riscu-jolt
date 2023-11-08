use ff::PrimeField;
use num_bigint::BigUint;

pub fn into_bigint<F: PrimeField>(i: &F) -> BigUint {
    BigUint::from_bytes_le(i.to_repr().as_ref())
}

pub fn num_bits<F: PrimeField>(i: &F) -> u64 {
    into_bigint(i).bits()
}

// Code below is for ark_ff, not sure to remove for now

// // Type that usually wraps a reference to a field type and implements display where "0" is
// // written if the value is zero.  This is used to overcome the fact that `BigInt` from
// // `ark_ff` writes "" when the value is 0.
// struct DispF<'a, T: Display + Zero>(&'a T);

// impl<'a, T: Display + Zero> Display for DispF<'a, T> {
//     fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
//         if self.0.is_zero() {
//             write!(f, "0")
//         } else {
//             write!(f, "{}", self.0)
//         }
//     }
// }

// // Type that wraps a reference to a slice of field type and implements display which writes the
// // list of elements in decimal form.  This is used to overcome the fact that a slice can only
// // be Debug-formatted (with `{:?}`), and `BigInt` from `ark_ff` writes an array of 4 limbs,
// // which is not very nice.
// struct DispFSlice<'a, T: Display + Zero>(&'a [T]);

// impl<'a, T: Display + Zero> Display for DispFSlice<'a, T> {
//     fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
//         write!(f, "[")?;
//         for (i, x) in self.0.iter().enumerate() {
//             if i != 0 {
//                 write!(f, ", ")?;
//             }
//             write!(f, "{}", DispF(x))?;
//         }
//         write!(f, "]")
//     }
// }
