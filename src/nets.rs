use bytemuck::{self, Pod, Zeroable};
use goober::activation::Activation;
use std::fs;
use std::io::Write;
use std::ops::AddAssign;
use std::path::Path;

use crate::subnets::QAA;

// Workaround for error in how goober handles an activation such as SCReLU
#[derive(Clone, Copy)]
pub struct SCReLU;

impl Activation for SCReLU {
    fn activate(x: f32) -> f32 {
        let clamped = x.clamp(0.0, 1.0);
        clamped * clamped
    }

    fn derivative(x: f32) -> f32 {
        if 0.0 < x && x < 1.0 {
            2.0 * x.sqrt()
        } else {
            0.0
        }
    }
}

#[derive(Clone, Copy, Debug, Zeroable)]
#[repr(C)]
pub struct Accumulator<T, const H: usize> {
    pub vals: [T; H],
}

unsafe impl<T: Copy + Zeroable + 'static, const H: usize> Pod for Accumulator<T, H> {}

impl<T: AddAssign, const H: usize> Accumulator<T, H> {
    pub fn set<U: Copy>(&mut self, weights: &Accumulator<U, H>)
    where
        T: From<U>,
    {
        for (i, d) in self.vals.iter_mut().zip(&weights.vals) {
            *i += T::from(*d);
        }
    }
}

impl<const H: usize> Accumulator<i16, H> {
    pub fn dot_relu(&self, rhs: &Accumulator<i16, H>) -> f32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if std::arch::is_x86_feature_detected!("sse2") {
                return self.dot_relu_sse2(rhs);
            }
        }

        let mut result: i32 = 0;
        for (a, b) in self.vals.iter().zip(&rhs.vals) {
            result += relu(*a) * relu(*b);
        }
        result as f32 / QAA as f32
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    unsafe fn dot_relu_sse2(&self, rhs: &Accumulator<i16, H>) -> f32 {
        use core::arch::x86_64::*;
        let mut sum = _mm_setzero_si128();
        let zero = _mm_setzero_si128();
        let mut i = 0;
        while i + 8 <= H {
            let a = _mm_loadu_si128(self.vals.as_ptr().add(i) as *const __m128i);
            let b = _mm_loadu_si128(rhs.vals.as_ptr().add(i) as *const __m128i);
            let a = _mm_max_epi16(a, zero);
            let b = _mm_max_epi16(b, zero);
            let prod = _mm_madd_epi16(a, b);
            sum = _mm_add_epi32(sum, prod);
            i += 8;
        }
        let mut tmp = [0i32; 4];
        _mm_storeu_si128(tmp.as_mut_ptr() as *mut __m128i, sum);
        let mut result = tmp.iter().sum::<i32>();
        for j in i..H {
            result += relu(self.vals[j]) * relu(rhs.vals[j]);
        }
        result as f32 / QAA as f32
    }
}

pub fn relu<F>(x: F) -> i32
where
    i32: From<F>,
{
    i32::from(x).max(0)
}

pub fn screlu(x: i16, q: i32) -> i32 {
    let clamped = i32::from(x).clamp(0, q);
    clamped * clamped
}

pub fn q_i16(x: f32, q: i32) -> i16 {
    let quantized = x * q as f32;
    assert!(f32::from(i16::MIN) < quantized && quantized < f32::from(i16::MAX),);
    quantized as i16
}

pub fn q_i32(x: f32, q: i32) -> i32 {
    let quantized = x * q as f32;
    assert!((i32::MIN as f32) < quantized && quantized < i32::MAX as f32);
    quantized as i32
}

pub fn save_to_bin<T: Pod>(dir: &Path, file_name: &str, data: &T) {
    let mut file = fs::File::create(dir.join(file_name)).expect("Failed to create file");

    let slice = bytemuck::bytes_of(data);

    file.write_all(slice).unwrap();
}
