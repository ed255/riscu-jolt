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

impl<F: PrimeField, V: Var> Expr<F, V> {
    fn _simplify(self) -> Self {
        use Expr::*;
        match self {
            Pow(e, f) => match *e {
                Const(e) => Const(e.pow([f as u64, 0, 0, 0])),
                _ => {
                    let e = e._simplify();
                    Pow(Box::new(e), f)
                }
            },
            Neg(e) => {
                let e = e._simplify();
                match e {
                    Neg(ne) => *ne, // double negate concels itself
                    e => Neg(Box::new(e)),
                }
            }
            Const(f) => Const(f),
            Var(v) => Var(v),
            Sum(es) => {
                let mut xs: Vec<Expr<F, V>> = Vec::new();
                for x in es.into_iter().map(|x| x._simplify()) {
                    match x {
                        Sum(es) => xs.extend(es.into_iter()),
                        e => xs.push(e),
                    }
                }
                xs.sort();
                let mut c = F::zero();
                let mut tail = Vec::new();
                for x in xs {
                    match x {
                        Neg(e) => match *e {
                            Const(a) => c -= a,
                            a => tail.push(Neg(Box::new(a))),
                        },
                        Const(a) => c += a,
                        a => tail.push(a),
                    }
                }
                let mut r = if c.is_zero() { vec![] } else { vec![Const(c)] };
                r.extend(tail.into_iter());
                match r.len() {
                    0 => Const(F::zero()),
                    1 => r.swap_remove(0),
                    _ => Sum(r),
                }
            }
            Mul(es) => {
                // TODO: get Pow's out of Mul elements
                let mut xs: Vec<Expr<F, V>> = Vec::new();
                let mut neg = false;
                for x in es.into_iter().map(|x| x._simplify()) {
                    match x {
                        Neg(e) => {
                            neg ^= true;
                            match *e {
                                Mul(es) => xs.extend(es.into_iter()),
                                ne => xs.push(ne),
                            }
                        }
                        Mul(es) => xs.extend(es.into_iter()),
                        e => xs.push(e),
                    }
                }
                xs.sort();
                let mut c = F::one();
                let mut tail = Vec::new();
                for x in xs {
                    match x {
                        Const(a) => c *= a,
                        a => tail.push(a),
                    }
                }
                let mut r = if c.is_zero() {
                    return Const(F::zero());
                } else if c.is_one() {
                    vec![]
                } else {
                    vec![Const(c)]
                };
                r.extend(tail.into_iter());
                let m = if r.len() == 1 {
                    r.swap_remove(0)
                } else if r.len() == 0 {
                    Const(F::one())
                } else {
                    Mul(r)
                };
                if neg {
                    Neg(Box::new(m))
                } else {
                    m
                }
            }
        }
    }

    /// Simplify the expression in places where it can be partially evaluated
    pub fn simplify_move(self) -> Self {
        let e = self._simplify();
        // let e = e.normalize_linear_comb();
        // let e = e.normalize_pow();
        e
    }

    /// Simplify the expression in places where it can be partially evaluated
    pub fn simplify(&mut self) -> &mut Self {
        let e = mem::replace(self, Expr::default());
        let e = e.simplify_move();
        *self = e;
        self
    }

    /// Take a list of multiplicands and return a Mul expression whith the multiplication of
    /// coefficients evaluated
    fn _mul_const(xs: Vec<Expr<F, V>>) -> Expr<F, V> {
        use Expr::*;
        let mut mul_const = F::one();
        let mut mul_exprs = Vec::new();
        for x in xs {
            match x {
                Const(f) => mul_const *= f,
                e => mul_exprs.push(e),
            }
        }
        if mul_exprs.len() == 0 {
            return Const(mul_const);
        }
        let mut xs = Vec::new();
        if !mul_const.is_one() {
            xs.push(Const(mul_const))
        }
        xs.extend_from_slice(&mul_exprs[..]);
        Mul(xs)
    }

    /// Apply "a * b % p" where a and b are expressions.  Evaluate coefficient multiplication in
    /// the resulting expression.
    fn _normalize_mul(a: Expr<F, V>, b: Expr<F, V>) -> Expr<F, V> {
        use Expr::*;
        match (a, b) {
            (Const(a), Const(b)) => Const(a * b),
            (Mul(mut xs), Mul(ys)) => {
                xs.extend_from_slice(&ys[..]);
                Self::_mul_const(xs)
            }
            (e, Mul(xs)) => {
                let mut ys = vec![e];
                ys.extend_from_slice(&xs[..]);
                Self::_mul_const(ys)
            }
            (Mul(mut xs), e) => {
                xs.push(e.clone());
                Self::_mul_const(xs)
            }
            (a, b) => Mul(vec![a, b]),
        }
    }

    fn _normalize(self) -> Self {
        use Expr::*;
        // p-1 == -1
        let p_1 = F::zero() - F::one();
        match self {
            Neg(e) => Mul(vec![Const(p_1), *e])._normalize(),
            Sum(xs) => {
                let xs = xs.into_iter().map(|x: Expr<F, V>| x._normalize());
                let mut sum_const = F::zero();
                let mut sum_exprs = Vec::new();
                for x in xs {
                    match x {
                        Const(f) => sum_const += f,
                        Sum(xs) => {
                            for x in xs {
                                match x {
                                    Const(f) => sum_const += f,
                                    e => sum_exprs.push(e),
                                }
                            }
                        }
                        e => sum_exprs.push(e),
                    }
                }
                let mut xs = Vec::new();
                if !sum_const.is_zero() {
                    xs.push(Const(sum_const))
                }
                xs.extend_from_slice(&sum_exprs[..]);
                Sum(xs)
            }
            Mul(xs) => {
                // println!("DBG1 {}", Mul(xs.clone()));
                let xs = xs.into_iter().map(|x| x._normalize());
                // flat muls
                let mut ys = Vec::new();
                for x in xs {
                    match x {
                        Mul(xs) => {
                            ys.extend_from_slice(&xs[..]);
                        }
                        _ => ys.push(x),
                    }
                }
                let xs = ys;
                let mut mul_const = F::one();
                let mut mul_vars: Vec<Expr<F, V>> = Vec::new();
                let mut mul_sums: Vec<Vec<Expr<F, V>>> = Vec::new();

                let mut ys: Vec<Expr<F, V>> = Vec::new();
                // Flatten exponentiations
                for x in xs.into_iter() {
                    match x {
                        Pow(e, f) => (0..f).for_each(|_| ys.push(e.as_ref().clone())),
                        e => ys.push(e),
                    }
                }
                let xs = ys;

                for x in xs {
                    match x {
                        Const(f) => mul_const *= f,
                        Var(v) => mul_vars.push(Var(v)),
                        Sum(xs) => mul_sums.push(xs),
                        _ => {
                            unreachable!();
                        }
                    }
                }

                let mut first = Vec::new();
                if !mul_const.is_one() {
                    first.push(Const(mul_const))
                }
                first.extend_from_slice(&mul_vars[..]);
                while mul_sums.len() >= 2 {
                    let mut result = Vec::new();
                    let lhs = &mul_sums[mul_sums.len() - 1];
                    let rhs = &mul_sums[mul_sums.len() - 2];
                    for a in lhs {
                        for b in rhs {
                            result.push(Self::_normalize_mul(a.clone(), b.clone()));
                        }
                    }
                    mul_sums.pop();
                    let last_index = mul_sums.len() - 1;
                    mul_sums[last_index] = result;
                }
                if mul_sums.len() > 0 {
                    for e in mul_sums[0].iter_mut() {
                        *e = Self::_normalize_mul(Mul(first.clone()), e.clone());
                    }
                    // println!("DBG2 {}", Sum(mul_sums[0].clone()));
                    Sum(mul_sums.pop().unwrap())
                } else {
                    // println!("DBG3 {}", Mul(first.clone()));
                    Self::_mul_const(first)
                }
            }
            Pow(e, f) => {
                let e = e._normalize();
                match e {
                    Const(b) => Const(b.pow([f as u64, 0, 0, 0])),
                    e => Pow(Box::new(e), f),
                }
            }
            _ => self,
        }
    }

    /// Return the expression in coefficient form
    pub fn normalize_move(self) -> Self {
        self.simplify_move()._normalize()
    }

    /// Return the expression in coefficient form
    pub fn normalize(&mut self) -> &mut Self {
        let e = mem::replace(self, Expr::default());
        let e = e.normalize_move();
        *self = e;
        self
    }
}
