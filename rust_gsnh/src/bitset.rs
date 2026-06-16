//! Safe deterministic bitset utilities for future GSNH masks.
//!
//! The Python implementation remains the correctness oracle.  This module is a
//! small dependency-free layer for representing sample masks compactly as
//! `Vec<u64>` while keeping all boundary checks explicit.

/// Compact fixed-length bitset backed by `Vec<u64>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitSet {
    len: usize,
    words: Vec<u64>,
}

impl BitSet {
    /// Create an all-false bitset with `len` addressable bits.
    pub fn new(len: usize) -> Self {
        Self {
            len,
            words: vec![0; word_count(len)],
        }
    }

    /// Create an all-true bitset with `len` addressable bits.
    pub fn with_all(len: usize) -> Self {
        let mut out = Self {
            len,
            words: vec![u64::MAX; word_count(len)],
        };
        out.mask_padding_bits();
        out
    }

    /// Create a bitset by setting each index in `indices`.
    ///
    /// Duplicate indices are accepted and are idempotent.
    pub fn from_indices(len: usize, indices: &[usize]) -> Result<Self, String> {
        let mut out = Self::new(len);
        for &index in indices {
            out.set(index)?;
        }
        Ok(out)
    }

    /// Number of addressable bits.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return true when no addressable bit is set.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count_ones() == 0
    }

    /// Set one bit, returning an error if `index >= len`.
    pub fn set(&mut self, index: usize) -> Result<(), String> {
        let (word, mask) = self.word_and_mask(index)?;
        self.words[word] |= mask;
        Ok(())
    }

    /// Clear one bit, returning an error if `index >= len`.
    pub fn unset(&mut self, index: usize) -> Result<(), String> {
        let (word, mask) = self.word_and_mask(index)?;
        self.words[word] &= !mask;
        Ok(())
    }

    /// Test one bit, returning an error if `index >= len`.
    pub fn contains(&self, index: usize) -> Result<bool, String> {
        let (word, mask) = self.word_and_mask(index)?;
        Ok((self.words[word] & mask) != 0)
    }

    /// Count set addressable bits.  Padding bits are never counted.
    pub fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// Return sorted set indices in increasing order.
    pub fn indices(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.count_ones());
        for index in 0..self.len {
            // Safe because index is constructed in range.
            if self.contains(index).unwrap_or(false) {
                out.push(index);
            }
        }
        out
    }

    /// Bitwise union.  Both bitsets must have the same length.
    pub fn union(&self, other: &Self) -> Result<Self, String> {
        self.ensure_compatible(other)?;
        Ok(Self::from_words(
            self.len,
            self.words
                .iter()
                .zip(other.words.iter())
                .map(|(a, b)| a | b)
                .collect(),
        ))
    }

    /// Bitwise intersection.  Both bitsets must have the same length.
    pub fn intersection(&self, other: &Self) -> Result<Self, String> {
        self.ensure_compatible(other)?;
        Ok(Self::from_words(
            self.len,
            self.words
                .iter()
                .zip(other.words.iter())
                .map(|(a, b)| a & b)
                .collect(),
        ))
    }

    /// Set difference `self \ other`.  Both bitsets must have the same length.
    pub fn difference(&self, other: &Self) -> Result<Self, String> {
        self.ensure_compatible(other)?;
        Ok(Self::from_words(
            self.len,
            self.words
                .iter()
                .zip(other.words.iter())
                .map(|(a, b)| a & !b)
                .collect(),
        ))
    }

    /// Complement all addressable bits.  Padding bits beyond `len` stay hidden.
    pub fn complement(&self) -> Self {
        Self::from_words(self.len, self.words.iter().map(|word| !word).collect())
    }

    /// Return true iff every set bit in `self` is also set in `other`.
    pub fn is_subset_of(&self, other: &Self) -> Result<bool, String> {
        self.ensure_compatible(other)?;
        Ok(self
            .words
            .iter()
            .zip(other.words.iter())
            .all(|(a, b)| (a & !b) == 0))
    }

    /// Return true iff the two bitsets share at least one set bit.
    pub fn intersects(&self, other: &Self) -> Result<bool, String> {
        self.ensure_compatible(other)?;
        Ok(self
            .words
            .iter()
            .zip(other.words.iter())
            .any(|(a, b)| (a & b) != 0))
    }

    fn from_words(len: usize, words: Vec<u64>) -> Self {
        debug_assert_eq!(words.len(), word_count(len));
        let mut out = Self { len, words };
        out.mask_padding_bits();
        out
    }

    fn ensure_compatible(&self, other: &Self) -> Result<(), String> {
        if self.len != other.len {
            Err(format!(
                "bitset length mismatch: left len {} != right len {}",
                self.len, other.len
            ))
        } else {
            Ok(())
        }
    }

    fn word_and_mask(&self, index: usize) -> Result<(usize, u64), String> {
        if index >= self.len {
            return Err(format!(
                "bit index {index} out of range for bitset length {}",
                self.len
            ));
        }
        Ok((index / 64, 1u64 << (index % 64)))
    }

    fn mask_padding_bits(&mut self) {
        let Some(last) = self.words.last_mut() else {
            return;
        };
        let remainder = self.len % 64;
        if remainder != 0 {
            let mask = (1u64 << remainder) - 1;
            *last &= mask;
        }
    }
}

#[inline]
fn word_count(len: usize) -> usize {
    (len + 63) / 64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_bitset_has_no_bits() {
        let bs = BitSet::new(0);
        assert_eq!(bs.len(), 0);
        assert!(bs.is_empty());
        assert_eq!(bs.count_ones(), 0);
        assert_eq!(bs.indices(), Vec::<usize>::new());
        assert!(bs.contains(0).is_err());
        assert_eq!(bs.complement(), BitSet::new(0));
    }

    #[test]
    fn single_bit_set_unset_contains() {
        let mut bs = BitSet::new(3);
        assert!(bs.is_empty());
        assert_eq!(bs.contains(1), Ok(false));

        bs.set(1).unwrap();
        assert!(!bs.is_empty());
        assert_eq!(bs.contains(1), Ok(true));
        assert_eq!(bs.count_ones(), 1);
        assert_eq!(bs.indices(), vec![1]);

        bs.unset(1).unwrap();
        assert_eq!(bs.contains(1), Ok(false));
        assert!(bs.is_empty());
    }

    #[test]
    fn out_of_range_operations_return_errors() {
        let mut bs = BitSet::new(2);
        assert!(bs.set(2).is_err());
        assert!(bs.unset(2).is_err());
        assert!(bs.contains(2).is_err());
        assert_eq!(bs.count_ones(), 0);
    }

    #[test]
    fn from_indices_accepts_duplicates() {
        let bs = BitSet::from_indices(6, &[4, 1, 4, 2, 1]).unwrap();
        assert_eq!(bs.count_ones(), 3);
        assert_eq!(bs.indices(), vec![1, 2, 4]);
    }

    #[test]
    fn from_indices_rejects_out_of_range() {
        let err = BitSet::from_indices(3, &[0, 3]).unwrap_err();
        assert!(err.contains("out of range"));
    }

    #[test]
    fn indices_are_sorted() {
        let bs = BitSet::from_indices(10, &[9, 0, 5, 3]).unwrap();
        assert_eq!(bs.indices(), vec![0, 3, 5, 9]);
    }

    #[test]
    fn set_operations_work() {
        let a = BitSet::from_indices(8, &[0, 2, 4, 6]).unwrap();
        let b = BitSet::from_indices(8, &[1, 2, 3, 6]).unwrap();

        assert_eq!(a.union(&b).unwrap().indices(), vec![0, 1, 2, 3, 4, 6]);
        assert_eq!(a.intersection(&b).unwrap().indices(), vec![2, 6]);
        assert_eq!(a.difference(&b).unwrap().indices(), vec![0, 4]);
    }

    #[test]
    fn complement_masks_padding_for_non_multiple_of_64() {
        let bs = BitSet::from_indices(70, &[0, 63, 69]).unwrap();
        let comp = bs.complement();
        assert_eq!(comp.len(), 70);
        assert_eq!(comp.count_ones(), 67);
        assert_eq!(comp.contains(0), Ok(false));
        assert_eq!(comp.contains(63), Ok(false));
        assert_eq!(comp.contains(69), Ok(false));
        assert!(comp.contains(70).is_err());

        let all = BitSet::with_all(70);
        assert_eq!(all.count_ones(), 70);
        assert_eq!(all.indices().last().copied(), Some(69));
    }

    #[test]
    fn subset_and_intersects_work() {
        let a = BitSet::from_indices(5, &[1, 3]).unwrap();
        let b = BitSet::from_indices(5, &[0, 1, 2, 3]).unwrap();
        let c = BitSet::from_indices(5, &[4]).unwrap();

        assert_eq!(a.is_subset_of(&b), Ok(true));
        assert_eq!(b.is_subset_of(&a), Ok(false));
        assert_eq!(a.intersects(&b), Ok(true));
        assert_eq!(a.intersects(&c), Ok(false));
    }

    #[test]
    fn incompatible_lengths_are_errors() {
        let a = BitSet::new(3);
        let b = BitSet::new(4);

        assert!(a.union(&b).is_err());
        assert!(a.intersection(&b).is_err());
        assert!(a.difference(&b).is_err());
        assert!(a.is_subset_of(&b).is_err());
        assert!(a.intersects(&b).is_err());
    }

    #[test]
    fn deterministic_equivalence_against_vec_bool_reference() {
        let len = 70;
        let left_indices = [0, 1, 2, 63, 65, 69];
        let right_indices = [1, 3, 63, 64, 69];
        let left = BitSet::from_indices(len, &left_indices).unwrap();
        let right = BitSet::from_indices(len, &right_indices).unwrap();

        let mut left_ref = vec![false; len];
        let mut right_ref = vec![false; len];
        for &idx in &left_indices {
            left_ref[idx] = true;
        }
        for &idx in &right_indices {
            right_ref[idx] = true;
        }

        let to_indices = |mask: &[bool]| -> Vec<usize> {
            mask.iter()
                .enumerate()
                .filter_map(|(idx, value)| value.then_some(idx))
                .collect()
        };

        let union_ref: Vec<bool> = left_ref
            .iter()
            .zip(right_ref.iter())
            .map(|(a, b)| *a || *b)
            .collect();
        let intersection_ref: Vec<bool> = left_ref
            .iter()
            .zip(right_ref.iter())
            .map(|(a, b)| *a && *b)
            .collect();
        let difference_ref: Vec<bool> = left_ref
            .iter()
            .zip(right_ref.iter())
            .map(|(a, b)| *a && !*b)
            .collect();
        let complement_ref: Vec<bool> = left_ref.iter().map(|value| !*value).collect();

        assert_eq!(left.indices(), to_indices(&left_ref));
        assert_eq!(right.indices(), to_indices(&right_ref));
        assert_eq!(
            left.union(&right).unwrap().indices(),
            to_indices(&union_ref)
        );
        assert_eq!(
            left.intersection(&right).unwrap().indices(),
            to_indices(&intersection_ref)
        );
        assert_eq!(
            left.difference(&right).unwrap().indices(),
            to_indices(&difference_ref)
        );
        assert_eq!(left.complement().indices(), to_indices(&complement_ref));
        assert_eq!(
            left.intersects(&right).unwrap(),
            left_ref.iter().zip(right_ref.iter()).any(|(a, b)| *a && *b)
        );
    }
}
