// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-side multimodal media preprocessing.
//!
//! Wave 1: shared image decode plus the Gemma-4 E2B image preprocessor
//! (resize → rescale → patchify → position grid) and the audio mel-spectrogram
//! frontend (`mel`) consumed by the E2B audio tower in a later wave.

pub mod decode;
pub mod gemma_audio;
pub mod gemma_vision;
pub mod mel;
