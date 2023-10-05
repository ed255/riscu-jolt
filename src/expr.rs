use ark_ff::{biginteger::BigInteger, One, PrimeField, Zero};
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};
use std::{
    cmp::{Eq, Ord, Ordering, PartialEq},
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

impl<F: PrimeField, V: Var> Ord for Expr<F, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        use Expr::*;
        use Ordering::*;
        match (self, other) {
            (Const(_), Const(_)) => Equal,
            (Const(_), Var(_)) => Less,
            (Const(_), Sum(_)) => Less,
            (Const(_), Mul(_)) => Less,
            (Const(_), Neg(e)) => self.cmp(e),
            (Var(_), Const(_)) => Greater,
            (Var(_), Var(_)) => Equal, // TODO
            (Var(_), Sum(_)) => Less,
            (Var(_), Mul(_)) => Less,
            (Var(_), Neg(e)) => self.cmp(e),
            (Sum(_), Const(_)) => Greater,
            (Sum(_), Var(_)) => Greater,
            (Sum(a), Sum(b)) => a.len().cmp(&b.len()),
            (Sum(_), Mul(_)) => Less,
            (Sum(_), Neg(e)) => self.cmp(e),
            (Mul(_), Const(_)) => Greater,
            (Mul(_), Var(_)) => Greater,
            (Mul(_), Sum(_)) => Greater,
            (Mul(a), Mul(b)) => a.len().cmp(&b.len()),
            (Mul(_), Neg(e)) => self.cmp(e),
            (Neg(e), Const(_)) => (**e).cmp(other),
            (Neg(e), Var(_)) => (**e).cmp(other),
            (Neg(e), Sum(_)) => (**e).cmp(other),
            (Neg(e), Mul(_)) => (**e).cmp(other),
            (Neg(e1), Neg(e2)) => (**e1).cmp(e2),
            _ => Equal,
        }
    }
}

impl<F: PrimeField, V: Var> Eq for Expr<F, V> {}

impl<F: PrimeField, V: Var> PartialOrd for Expr<F, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
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
    let pow2 = if c_bits == 0 {
        F::one()
    } else {
        F::from(2u64).pow([c_bits as u64 - 1, 0, 0, 0])
    };
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

#[derive(Debug, Clone)]
pub struct Term<F: PrimeField, V: Var> {
    pub coeff: F,
    pub vars: Vec<(V, u32)>,
}

#[derive(Debug, Clone)]
pub struct ExprTerms<F: PrimeField, V: Var>(Vec<Term<F, V>>);

impl<F: PrimeField, V: Var + Display> Display for ExprTerms<F, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let neg = F::zero() - F::one();
        for (i, term) in self.0.iter().enumerate() {
            let neg_coeff = term.coeff * neg;
            if (neg_coeff).into_bigint().num_bits() < 100 {
                write!(f, " - {}", neg_coeff)?;
            } else {
                if i != 0 {
                    write!(f, " + ")?;
                }
                if !term.coeff.is_one() {
                    write!(f, "{}", term.coeff)?;
                }
            }
            for (j, var) in term.vars.iter().enumerate() {
                if j != 0 || (j == 0 && !term.coeff.is_one()) {
                    write!(f, "*")?;
                }
                write!(f, "{}", var.0)?;
                if var.1 > 1 {
                    write!(f, "^{}", var.1)?;
                }
            }
        }
        Ok(())
    }
}

// Normalization implementation.  Here normalization means transorming the polynomial expression
// into the equivalent coefficient form which consists of a sum of terms where each term has
// an integer coefficient and a list of variables raised to some positive integer power.
impl<F: PrimeField, V: Var> Expr<F, V> {
    // Multiply two normalized expressions and return the normalized result
    fn _mul2_normalize(lhs: Vec<Term<F, V>>, rhs: Vec<Term<F, V>>) -> Vec<Term<F, V>> {
        let mut terms = Vec::new();
        for term_lhs in &lhs {
            for term_rhs in &rhs {
                let mut vars: Vec<(V, u32)> = Vec::new();
                let (mut lhs_i, mut rhs_i) = (0, 0);
                loop {
                    let (lhs, rhs) = (term_lhs.vars.get(lhs_i), term_rhs.vars.get(rhs_i));
                    match (lhs, rhs) {
                        (Some((lhs_var, lhs_exp)), Some((rhs_var, rhs_exp))) => {
                            if lhs_var < rhs_var {
                                lhs_i += 1;
                                vars.push((lhs_var.clone(), *lhs_exp));
                            } else if rhs_var < lhs_var {
                                rhs_i += 1;
                                vars.push((rhs_var.clone(), *rhs_exp));
                            } else {
                                // Merge
                                lhs_i += 1;
                                rhs_i += 1;
                                vars.push((lhs_var.clone(), *lhs_exp + *rhs_exp));
                            }
                        }
                        (Some(lhs), None) => {
                            lhs_i += 1;
                            vars.push(lhs.clone());
                        }
                        (None, Some(rhs)) => {
                            rhs_i += 1;
                            vars.push(rhs.clone())
                        }
                        (None, None) => break,
                    }
                }
                terms.push(Term {
                    coeff: term_lhs.coeff * term_rhs.coeff,
                    vars,
                })
            }
        }
        terms
    }

    fn _normalize(&self) -> Vec<Term<F, V>> {
        use Expr::*;
        // p-1 == -1
        let p_1 = F::zero() - F::one();
        match self {
            Neg(e) => {
                let mut terms = e._normalize();
                terms.iter_mut().for_each(|term| term.coeff *= p_1);
                terms
            }
            Sum(xs) => {
                let terms = xs.iter().map(|x: &Expr<F, V>| x._normalize()).flatten();
                let mut sum_const = F::zero();
                let mut sum_terms = Vec::new();
                for Term { coeff, vars } in terms {
                    if vars.len() == 0 {
                        sum_const += coeff
                    } else {
                        sum_terms.push(Term { coeff, vars });
                    }
                }
                let mut terms = Vec::new();
                if !sum_const.is_zero() {
                    terms.push(Term {
                        coeff: sum_const,
                        vars: vec![],
                    });
                }
                terms.extend_from_slice(&sum_terms[..]);
                terms
            }
            Mul(xs) => {
                let mut terms2 = xs.into_iter().map(|x| x._normalize());
                if let Some(mut terms) = terms2.next() {
                    for terms_next in terms2 {
                        terms = Self::_mul2_normalize(terms, terms_next);
                    }
                    terms
                } else {
                    Vec::new()
                }
            }
            Pow(e, f) => {
                let pow_terms = e._normalize();
                if pow_terms.len() == 1 && pow_terms[0].vars.len() == 0 {
                    vec![Term {
                        coeff: pow_terms[0].coeff.pow([*f as u64, 0, 0, 0]),
                        vars: vec![],
                    }]
                } else {
                    let mut terms = pow_terms.clone();
                    for i in 1..*f {
                        terms = Self::_mul2_normalize(terms, pow_terms.clone());
                    }
                    terms
                }
            }
            Var(v) => vec![Term {
                coeff: F::one(),
                vars: vec![(v.clone(), 1)],
            }],
            Const(c) => vec![Term {
                coeff: *c,
                vars: vec![],
            }],
        }
    }

    /// Return the expression in coefficient form
    pub fn normalize(&self) -> ExprTerms<F, V> {
        let mut terms = self._normalize();
        // Group terms with same vars
        terms.sort_by(|lhs_term, rhs_term| lhs_term.vars.cmp(&rhs_term.vars));
        let mut terms_iter = terms.into_iter();
        if let Some(first) = terms_iter.next() {
            let mut grouped_terms = vec![first];
            for term in terms_iter {
                let last = grouped_terms.last_mut().expect("has len > 0");
                if last.vars == term.vars {
                    last.coeff += term.coeff;
                } else {
                    grouped_terms.push(term)
                }
            }
            ExprTerms(grouped_terms)
        } else {
            ExprTerms(vec![])
        }
    }
}
