#![allow(dead_code)]
#![allow(unused_variables)] // TODO: Remove this in the future
pub mod emulator;
mod expr;
pub mod simulator;
#[cfg(test)]
mod tests;

use std::ops::{Index, IndexMut};

const REG_SP: usize = 2;

#[derive(Default)]
pub struct Registers<T> {
    r: [T; 32],
    zero: T,
}

impl<T> Index<usize> for Registers<T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        &self.r[i]
    }
}

impl<T> IndexMut<usize> for Registers<T> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        if i == 0 {
            &mut self.zero
        } else {
            &mut self.r[i]
        }
    }
}
