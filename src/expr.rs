use ark_ff::{biginteger::BigInteger, One, PrimeField, Zero};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};
use std::{
    cmp::{Eq, Ord, PartialEq},
    fmt::{self, Debug, Display, Write},
    hash::Hash,
    mem,
};

pub trait Arithmetic:
    // 'static
    Clone
    + Debug
    + Display
    + Default
    // + Send
    // + Sync
    // + Eq
    + Zero
    + One
    + Neg<Output = Self>
    // + Hash
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    // + Div<Self, Output = Self>
    + AddAssign<Self>
    // + SubAssign<Self>
    + MulAssign<Self>
    // + DivAssign<Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    // + for<'a> Div<&'a Self, Output = Self>
    // + for<'a> AddAssign<&'a Self>
    // + for<'a> SubAssign<&'a Self>
    // + for<'a> MulAssign<&'a Self>
    // + for<'a> DivAssign<&'a Self>
    // + for<'a> Add<&'a mut Self, Output = Self>
    // + for<'a> Sub<&'a mut Self, Output = Self>
    // + for<'a> Mul<&'a mut Self, Output = Self>
    // + for<'a> Div<&'a mut Self, Output = Self>
    // + for<'a> AddAssign<&'a mut Self>
    // + for<'a> SubAssign<&'a mut Self>
    // + for<'a> MulAssign<&'a mut Self>
    // + for<'a> DivAssign<&'a mut Self>
    + From<u128>
    + From<u64>
    // + From<u32>
    // + From<u16>
    // + From<u8>
    // + From<bool>
{
}

impl<T: PrimeField> Arithmetic for T {}

pub trait Var: Clone + Debug + PartialEq + Eq + Hash + Ord + Display {}

impl Var for &'static str {}
impl Var for String {}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr<F: PrimeField, V: Var> {
    Const(F),
    Var(V),
    Sum(Vec<Expr<F, V>>),
    Mul(Vec<Expr<F, V>>),
    Neg(Box<Expr<F, V>>),
    Pow(Box<Expr<F, V>>, u32),
}

impl<F: PrimeField, V: Var> Default for Expr<F, V> {
    fn default() -> Self {
        Self::Const(F::zero())
    }
}

impl<F: PrimeField, V: Var> From<u64> for Expr<F, V> {
    fn from(c: u64) -> Self {
        Self::Const(F::from(c))
    }
}

impl<F: PrimeField, V: Var> From<u128> for Expr<F, V> {
    fn from(c: u128) -> Self {
        Self::Const(F::from(c))
    }
}

impl<F: PrimeField, V: Var> Zero for Expr<F, V> {
    fn zero() -> Self {
        Self::Const(F::zero())
    }

    fn is_zero(&self) -> bool {
        match self {
            Self::Const(c) => c.is_zero(),
            _ => false,
        }
    }
}

impl<F: PrimeField, V: Var> One for Expr<F, V> {
    fn one() -> Self {
        Self::Const(F::one())
    }

    fn is_one(&self) -> bool {
        match self {
            Self::Const(c) => c.is_one(),
            _ => false,
        }
    }
}

impl<F: PrimeField, V: Var> Arithmetic for Expr<F, V> {}

impl<F: PrimeField, V: Var> Add for Expr<F, V> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        use Expr::*;
        match self {
            Sum(mut xs) => {
                xs.push(rhs);
                Sum(xs)
            }
            e => Sum(vec![e, rhs]),
        }
    }
}

impl<'a, F: PrimeField, V: Var> Add<&'a Expr<F, V>> for Expr<F, V> {
    type Output = Self;
    fn add(self, rhs: &Self) -> Self {
        use Expr::*;
        match self {
            Sum(mut xs) => {
                xs.push(rhs.clone());
                Sum(xs)
            }
            e => Sum(vec![e, rhs.clone()]),
        }
    }
}

impl<F: PrimeField, V: Var> AddAssign for Expr<F, V> {
    fn add_assign(&mut self, other: Self) {
        let this = mem::take(self);
        *self = this + other;
    }
}

impl<F: PrimeField, V: Var> Sub for Expr<F, V> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        use Expr::*;
        match self {
            Sum(mut xs) => {
                xs.push(rhs.neg());
                Sum(xs)
            }
            e => Sum(vec![e, rhs.neg()]),
        }
    }
}

impl<'a, F: PrimeField, V: Var> Sub<&'a Expr<F, V>> for Expr<F, V> {
    type Output = Self;
    fn sub(self, rhs: &Self) -> Self {
        use Expr::*;
        match self {
            Sum(mut xs) => {
                xs.push(rhs.clone().neg());
                Sum(xs)
            }
            e => Sum(vec![e, rhs.clone().neg()]),
        }
    }
}

impl<F: PrimeField, V: Var> Mul for Expr<F, V> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        use Expr::*;
        match self {
            Mul(mut xs) => {
                xs.push(rhs);
                Mul(xs)
            }
            e => Mul(vec![e, rhs]),
        }
    }
}

impl<'a, F: PrimeField, V: Var> Mul<&'a Expr<F, V>> for Expr<F, V> {
    type Output = Self;
    fn mul(self, rhs: &Self) -> Self {
        use Expr::*;
        match self {
            Mul(mut xs) => {
                xs.push(rhs.clone());
                Mul(xs)
            }
            e => Mul(vec![e, rhs.clone()]),
        }
    }
}

impl<F: PrimeField, V: Var> MulAssign for Expr<F, V> {
    fn mul_assign(&mut self, other: Self) {
        let this = mem::take(self);
        *self = this * other;
    }
}

impl<F: PrimeField, V: Var> Neg for Expr<F, V> {
    type Output = Self;
    fn neg(self) -> Self {
        Expr::Neg(Box::new(self))
    }
}

pub struct ExprDisplay<'a, F: PrimeField, V: Var, T>
where
    T: Fn(&mut fmt::Formatter<'_>, &V) -> fmt::Result,
{
    pub e: &'a Expr<F, V>,
    pub var_fmt: T,
}

impl<'a, F: PrimeField, V: Var, T> Display for ExprDisplay<'a, F, V, T>
where
    T: Fn(&mut fmt::Formatter<'_>, &V) -> fmt::Result,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.e.fmt_ascii(f, &self.var_fmt)
    }
}

impl<F: PrimeField, V: Var + Display> Display for Expr<F, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_ascii(f, &mut |f: &mut fmt::Formatter<'_>, v: &V| {
            write!(f, "{}", v)
        })
    }
}

pub(crate) fn fmt_f<W: Write, F: PrimeField>(f: &mut W, c: &F) -> fmt::Result {
    let c_bi = c.into_bigint();
    let c_bits = c_bi.num_bits();
    let pow2 = F::from(2u64).pow([c_bits as u64 - 1, 0, 0, 0]);
    if c_bits >= 8 && c == &pow2 {
        write!(f, "2^{}", c_bits - 1)
        // TODO: Implement hex format
        // } else if c_bits >= 16 {
        //     write!(f, "0x{:x}", c_bi)
    } else {
        write!(f, "{}", c_bi)
    }
}

impl<F: PrimeField, V: Var> Expr<F, V> {
    // sumatory terminal
    fn is_terminal(&self) -> bool {
        matches!(self, Expr::Const(_) | Expr::Var(_) | Expr::Pow(_, _))
    }

    // multiplicatory terminal
    fn is_mul_terminal(&self) -> bool {
        self.is_terminal() || matches!(self, Expr::Mul(_))
    }

    pub fn fmt_ascii<W: Write, FV>(&self, f: &mut W, fmt_var: &FV) -> fmt::Result
    where
        FV: Fn(&mut W, &V) -> fmt::Result,
    {
        use Expr::*;
        let fmt_exp = |e: &Self, f: &mut W, parens: bool| -> fmt::Result {
            if parens {
                write!(f, "(")?;
            }
            e.fmt_ascii(f, fmt_var)?;
            if parens {
                write!(f, ")")?;
            }
            Ok(())
        };
        match self {
            Neg(e) => {
                write!(f, "-")?;
                let parens = !e.is_terminal();
                fmt_exp(e, f, parens)?;
                Ok(())
            }
            Pow(e, c) => {
                let parens = !e.is_terminal();
                fmt_exp(e, f, parens)?;
                write!(f, "^{}", c)
            }
            Const(c) => fmt_f(f, c),
            Var(v) => fmt_var(f, v),
            Sum(es) => {
                for (i, e) in es.iter().enumerate() {
                    let (neg, e) = if let Neg(e) = e {
                        (true, &**e)
                    } else {
                        (false, e)
                    };
                    if i == 0 {
                        if neg {
                            write!(f, "-")?;
                        }
                    } else if neg {
                        write!(f, " - ")?;
                    } else {
                        write!(f, " + ")?;
                    }
                    let parens = !e.is_mul_terminal();
                    fmt_exp(e, f, parens)?;
                }
                Ok(())
            }
            Mul(es) => {
                for (i, e) in es.iter().enumerate() {
                    let parens = !e.is_terminal();
                    fmt_exp(e, f, parens)?;
                    if i != es.len() - 1 {
                        write!(f, "*")?;
                    }
                }
                Ok(())
            }
        }
    }
}
