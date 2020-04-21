use std::{mem::MaybeUninit};

// use hashbrown::HashMap;
use hibitset::BitSetLike;

use crate::{
    storage::{DistinctStorage, UnprotectedStorage},
    world::Index,
};
use super::SliceAccess;

use metalgear::GPUVec;

// pub struct GPUVecStorage<T: Copy> {
//     data: GPUVec<T>,
//     entity_id: Vec<Index>,
//     data_id: Vec<MaybeUninit<Index>>,
// }

// impl<T: Copy> Default for GPUVecStorage<T> {
//     fn default() -> Self {
//         Self {
//             data: Default::default(),
//             entity_id: Default::default(),
//             data_id: Default::default(),
//         }
//     }
// }

// impl<T: Copy> SliceAccess<T> for GPUVecStorage<T> {
//     type Element = T;

//     /// Returns a slice of all the components in this storage.
//     ///
//     /// Indices inside the slice do not correspond to anything in particular,
//     /// and especially do not correspond with entity IDs.
//     #[inline]
//     fn as_slice(&self) -> &[Self::Element] {
//         self.data.as_slice()
//     }

//     /// Returns a mutable slice of all the components in this storage.
//     ///
//     /// Indices inside the slice do not correspond to anything in particular,
//     /// and especially do not correspond with entity IDs.
//     #[inline]
//     fn as_mut_slice(&mut self) -> &mut [Self::Element] {
//         self.data.as_mut_slice()
//     }
// }

// impl<T: Copy> UnprotectedStorage<T> for GPUVecStorage<T> {
//     unsafe fn clean<B>(&mut self, has: B)
//     where
//         B: BitSetLike,
//     {
//         use std::ptr;
//         for (i, v) in self.data.iter_mut().enumerate() {
//             if has.contains(i as u32) {
//                 // drop in place
//                 ptr::drop_in_place(&mut *v.as_mut_ptr());
//             }
//         }
//         self.data.set_len(0);
//     }

//     unsafe fn get(&self, id: Index) -> &T {
//         &*self.data.get_unchecked(id as usize).as_ptr()
//     }

//     unsafe fn get_mut(&mut self, id: Index) -> &mut T {
//         &mut *self.data.get_unchecked_mut(id as usize).as_mut_ptr()
//     }

//     unsafe fn insert(&mut self, id: Index, v: T) {
//         let id = id as usize;
//         if self.data.len() <= id {
//             let delta = id + 1 - self.data.len();
//             self.data.reserve(delta);
//             self.data.set_len(id + 1);
//         }
//         // Write the value without reading or dropping
//         // the (currently uninitialized) memory.
//         *self.data.get_unchecked_mut(id as usize) = MaybeUninit::new(v);
//     }

//     unsafe fn remove(&mut self, id: Index) -> T {
//         use std::ptr;
//         ptr::read(self.get(id))
//     }
// }

// unsafe impl<T: Copy> DistinctStorage for GPUVecStorage<T> {}

/// Like `DenseVecStorage` but on the GPU
pub struct GPUDenseVecStorage<T: Copy> {
    data: GPUVec<T>,
    entity_id: Vec<Index>,
    data_id: Vec<MaybeUninit<Index>>,
}

impl<T: Copy> Default for GPUDenseVecStorage<T> {
    fn default() -> Self {
        Self {
            data: Default::default(),
            entity_id: Default::default(),
            data_id: Default::default(),
        }
    }
}

impl<T: Copy> SliceAccess<T> for GPUDenseVecStorage<T> {
    type Element = T;

    /// Returns a slice of all the components in this storage.
    ///
    /// Indices inside the slice do not correspond to anything in particular,
    /// and especially do not correspond with entity IDs.
    #[inline]
    fn as_slice(&self) -> &[Self::Element] {
        self.data.as_slice()
    }

    /// Returns a mutable slice of all the components in this storage.
    ///
    /// Indices inside the slice do not correspond to anything in particular,
    /// and especially do not correspond with entity IDs.
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [Self::Element] {
        self.data.as_mut_slice()
    }
}

impl<T: Copy> UnprotectedStorage<T> for GPUDenseVecStorage<T> {
    unsafe fn clean<B>(&mut self, _has: B)
    where
        B: BitSetLike,
    {
        // nothing to do
    }

    unsafe fn get(&self, id: Index) -> &T {
        let did = self.data_id.get_unchecked(id as usize).assume_init();
        self.data.get_unchecked(did as usize)
    }

    unsafe fn get_mut(&mut self, id: Index) -> &mut T {
        let did = self.data_id.get_unchecked(id as usize).assume_init();
        self.data.get_unchecked_mut(did as usize)
    }

    unsafe fn insert(&mut self, id: Index, v: T) {
        let id = id as usize;
        if self.data_id.len() <= id {
            let delta = id + 1 - self.data_id.len();
            self.data_id.reserve(delta);
            self.data_id.set_len(id + 1);
        }
        self.data_id
            .get_unchecked_mut(id)
            .as_mut_ptr()
            .write(self.data.len() as Index);
        self.entity_id.push(id as Index);
        self.data.push(v);
    }

    unsafe fn remove(&mut self, id: Index) -> T {
        let did = self.data_id.get_unchecked(id as usize).assume_init();
        let last = *self.entity_id.last().unwrap();
        self.data_id
            .get_unchecked_mut(last as usize)
            .as_mut_ptr()
            .write(did);
        self.entity_id.swap_remove(did as usize);
        self.data.swap_remove(did as usize)
    }
}

unsafe impl<T: Copy> DistinctStorage for GPUDenseVecStorage<T> {}

