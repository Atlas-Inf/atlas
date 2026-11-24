// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#define ATLAS_KV_ROW_SIZE(heads, dim) ((heads) * 128u + 64u)
#define ATLAS_KV_COLUMN(head, dim, col) ((col) < 128u ? (head) * 128u + (col) : 32u * 128u + (col) - 128u)
