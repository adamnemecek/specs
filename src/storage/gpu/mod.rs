use std::mem::MaybeUninit;

// use hashbrown::HashMap;
use hibitset::BitSetLike;

use crate::{
    storage::{DistinctStorage, UnprotectedStorage, Storage},
    world::Index,
};
use super::SliceAccess;
// use std::ops::DerefMut;

use metalgear::GPUVec;

mod gpuvecstorage;

pub trait GPUStorage<T: Copy> {
    fn gpu_index(&self) -> &GPUVec<MaybeUninit<Index>>;
    fn gpu_data(&self) -> &GPUVec<T>;
}

/// Like `DenseVecStorage` but on the GPU
pub struct GPUDenseVecStorage<T: Copy> {
    data: GPUVec<T>,
    entity_id: Vec<Index>,
    index: GPUVec<MaybeUninit<Index>>,
}



impl<T: Copy> GPUStorage<T> for GPUDenseVecStorage<T> {
    /// docs
    fn gpu_index(&self) -> &GPUVec<MaybeUninit<Index>> {
        &self.index
    }

    // docs
    // * you probably should not be accessing these mutably as that
    //      fuck with the inner structure
    // pub fn gpu_index_mut(&mut self) -> &mut GPUVec<MaybeUninit<Index>> {
    //     &mut self.index
    // }

    /// docs
    fn gpu_data(&self) -> &GPUVec<T> {
        &self.data
    }

    // docs
    // pub fn gpu_data_mut(&mut self) -> &mut GPUVec<T> {
    //     &mut self.data
    // }
}

// use metalgear::GPUResource;

// fn test(vec: GPUVec<u32>, encoder: &metal::RenderCommandEncoderRef) {
//     let d = vec.device();
//     let queue = d.new_command_queue();
//     let command_buffer = queue.new_command_buffer();

// }

// #[derive(Clone, Copy)]
// struct TestComp;

// impl crate::Component for TestComp {
//     type Storage = GPUDenseVecStorage<Self>;
// }

// struct TestSystem;

// impl<'a> crate::System<'a> for TestSystem {
//     type SystemData = (
//         crate::ReadStorage<'a, TestComp>
//     );

//     fn run(&mut self, data: Self::SystemData) {
//         let z = data.unprotected_storage();
//         let x = z.gpu_data();
//         // let z = data.inner.gpu_index();
//     }
// }

impl<T: Copy> Default for GPUDenseVecStorage<T> {
    fn default() -> Self {
        Self {
            data: Default::default(),
            entity_id: Default::default(),
            index: Default::default(),
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
        let did = self.index.get_unchecked(id as usize).assume_init();
        self.data.get_unchecked(did as usize)
    }

    unsafe fn get_mut(&mut self, id: Index) -> &mut T {
        let did = self.index.get_unchecked(id as usize).assume_init();
        self.data.get_unchecked_mut(did as usize)
    }

    unsafe fn insert(&mut self, id: Index, v: T) {
        let id = id as usize;
        if self.index.len() <= id {
            let delta = id + 1 - self.index.len();
            self.index.reserve(delta);
            self.index.set_len(id + 1);
        }
        self.index
            .get_unchecked_mut(id)
            .as_mut_ptr()
            .write(self.data.len() as Index);
        self.entity_id.push(id as Index);
        self.data.push(v);
    }

    unsafe fn remove(&mut self, id: Index) -> T {
        let did = self.index.get_unchecked(id as usize).assume_init();
        let last = *self.entity_id.last().unwrap();
        self.index
            .get_unchecked_mut(last as usize)
            .as_mut_ptr()
            .write(did);
        self.entity_id.swap_remove(did as usize);
        self.data.swap_remove(did as usize)
    }
}

unsafe impl<T: Copy> DistinctStorage for GPUDenseVecStorage<T> {}
