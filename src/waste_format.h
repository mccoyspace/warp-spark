/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 SQLite Cloud, Inc.
 */
/*
 * waste_format.h — WASTE container on-disk layout (format v0)
 *
 * See docs/FORMAT.md for the rationale. Sizes were frozen against the
 * released Kimi K3 weights (2026-07-27) and the containers built from
 * them; a reader must take the variable ones from the manifest rather
 * than from the comments here.
 *
 * Invariants:
 *   - every independently-readable record is 4 KiB aligned and sized in
 *     4 KiB multiples (O_DIRECT / F_NOCACHE safe);
 *   - an expert's gate/up/down live in ONE record → one pread();
 *   - all multi-byte fields little-endian; structs are packed, no padding
 *     surprises across compilers (static_asserts below).
 *
 * C11, no dependencies. This header defines layout only — no I/O.
 */

#ifndef WASTE_FORMAT_H
#define WASTE_FORMAT_H

#include <stdint.h>
#include <assert.h>

#define WASTE_FORMAT_VERSION 0

/* ---- magics ------------------------------------------------------------ */

#define WASTE_MAGIC_MANIFEST 0x45545357u /* "WSTE" — manifest.json header  */
#define WASTE_MAGIC_EXPERT   0x50584557u /* "WEXP" — expert record         */
#define WASTE_MAGIC_SUB      0x42555357u /* "WSUB" — substitute record     */
#define WASTE_MAGIC_TRUNK    0x4b525457u /* "WTRK" — trunk section         */
#define WASTE_MAGIC_CODEBOOK 0x4b424357u /* "WCBK" — codebook section      */
#define WASTE_MAGIC_LOWRANK  0x524c5757u /* "WWLR" — shared low-rank       */
#define WASTE_MAGIC_USAGE    0x47535557u /* "WUSG" — usage/hotlist log     */
#define WASTE_MAGIC_KDASTATE 0x53414457u /* "WDAS" — persisted KDA state   */

#define WASTE_ALIGN 4096u

/* ---- quantization formats (docs/FORMAT.md table) ----------------------- */

typedef enum {
    WQ_F32  = 0, /* norms, router, score-correction bias                   */
    WQ_F16  = 1, /* low-rank factors, codebooks (bf16 flagged via WF_BF16) */
    WQ_Q8G  = 2, /* int8, f16 scale per group of 128                       */
    WQ_Q4G  = 3, /* int4 packed, f16 scale per group of 128                */
    WQ_VQ3R = 4, /* 3.00 b/w: 3 residual stages, 8-dim vectors, 256/stage  */
    WQ_VQ2R = 5, /* 2.00 b/w: 2 residual stages, 8-dim vectors, 256/stage  */
    WQ_SUB1 = 6, /* ~1.0  b/w: direct VQ substitute (cache-miss fallback)  */
    WQ_Q3G  = 7, /* 3-bit packed, f16 scale per group of 128. Implemented
                  * end to end and not the default for anything: measured
                  * on K3's trunk it buys cache and loses more to the
                  * unpack, and the output collapses — docs/LEARNED.md §13 */
    WQ_VQ4P = 8, /* 3.00 b/w: 4 residual stages, 8-dim vectors, 64/stage,
                  * indices packed 4x6 bits into 3 bytes.
                  *
                  * Same bit rate as VQ3R and the same three bytes per
                  * vector — what changes is that a 64-entry stage table is
                  * 64 bytes, which is one NEON vqtbl4q, while VQ3R's
                  * 256-entry table is 16 vector registers and cannot be
                  * held in a register file that has 32. That is the whole
                  * reason the VQ3R gather is scalar.
                  *
                  * A distinct fmt rather than a manifest flag alone
                  * because the payload is byte-for-byte the same size as
                  * VQ3R's: a reader that took these three bytes for three
                  * one-byte indices would decode silently and wrongly,
                  * which is the failure a format code exists to prevent. */
} waste_fmt;

/* Bits per weight for VQxR is exactly `stages` — one byte of index per
 * 8-dim vector per stage — plus one f16 scale per output row, i.e.
 * 16/n_in amortized. VQ4P instead spends `index_bits` per stage, so its
 * rate is stages*index_bits/vec_dim: 4*6/8 = 3.00. The manifest's
 * expert_quant block carries the real values (`stages`, `vec_dim`,
 * `entries`, `index_bits`, `bits_per_weight`); nothing should hardcode
 * them. `index_bits` is absent in v0 containers and reads as 8. WQ_SUB1
 * is specified and not written by the converter in v0. */

/* record flags */
#define WF_BF16      (1u << 0) /* f16 payloads are bf16                    */
#define WF_HAS_SUB   (1u << 1) /* a WSUB record exists for this expert     */
#define WF_HOT_HINT  (1u << 2) /* converter-baked hotlist member           */

/* ---- expert bank: experts-L{layer}.bin --------------------------------- */
/*
 * File = contiguous 4 KiB-aligned ExpertRec, sorted by expert_id.
 * Offsets inside the record are relative to the record start, so a single
 * pread(fd, buf, rec_4k_blocks * 4096, file_off) yields a self-contained
 * expert; QT views point into the buffer (slab discipline).
 *
 * W ≈ (U_shared · V_sharedᵀ) ⊙ chan_correction + dequant(vq_indices)
 */

typedef struct {
    uint32_t magic;          /* WASTE_MAGIC_EXPERT                          */
    uint16_t layer;
    uint16_t expert_id;      /* 0..895                                      */
    uint8_t  fmt;            /* WQ_VQ3R | WQ_VQ2R                           */
    uint8_t  flags;          /* WF_*                                        */
    uint16_t codebook_id;    /* index into codebooks.bin                    */
    uint16_t lowrank_id;     /* MUST BE 0 in v0: the shared low-rank block
                              * is specified but not implemented — see
                              * docs/FORMAT.md "on probation"              */
    uint16_t reserved0;
    uint32_t rec_4k_blocks;  /* total record size / 4096                    */
    /* payload offsets, relative to record start (bytes)                    */
    uint32_t gate_off;
    uint32_t up_off;
    uint32_t down_off;
    uint32_t chan_corr_off;  /* per-channel f16 scale+bias, all 3 matrices  */
    uint32_t crc32;          /* payload checksum: zlib's crc32 over the
                              * record from the end of this header to the
                              * end of the per-channel scales, excluding
                              * the 4 KiB padding. Always written, and
                              * always checked by
                              * tools/verify_container.py. The engine
                              * checks it only when asked
                              * (waste_cfg.verify_records, WASTE_VERIFY=1):
                              * it is a pass over every record on every
                              * miss, ~5% of throughput, so it is the
                              * caller's call. What the engine always
                              * checks is this header, which the crc32
                              * does not cover — see model.c.            */
    uint32_t reserved1[2];
    /* payload follows, padded to 4 KiB multiple                            */
} waste_expert_hdr;

static_assert(sizeof(waste_expert_hdr) == 48, "expert header layout");

/* Substitute bank (subs-L{layer}.bin): same header with WASTE_MAGIC_SUB,
 * fmt = WQ_SUB1, ~5x smaller records, layout mirrors the expert bank.     */

/* ---- codebooks.bin ----------------------------------------------------- */

typedef struct {
    uint32_t magic;          /* WASTE_MAGIC_CODEBOOK                        */
    uint16_t codebook_id;
    uint8_t  fmt;            /* which WQ_* this codebook serves             */
    uint8_t  vec_dim;        /* 8                                           */
    uint32_t n_entries;      /* 256 per stage; the stage count is what
                              * VQ3R and VQ2R differ in, not this           */
    uint32_t reserved;
    /* n_entries * vec_dim f16 vectors follow                               */
} waste_codebook_hdr;

static_assert(sizeof(waste_codebook_hdr) == 16, "codebook header layout");

/* ---- lowrank.bin ------------------------------------------------------- */
/* Per (layer, matrix-kind) shared factors, FP16, always RAM-resident.
 * rank ≈ dim/128 (TBD): ~0.1 bit/param amortized across 896 experts.      */

typedef struct {
    uint32_t magic;          /* WASTE_MAGIC_LOWRANK                         */
    uint16_t lowrank_id;
    uint8_t  kind;           /* 0=gate 1=up 2=down                          */
    uint8_t  flags;
    uint16_t layer;
    uint16_t rank;
    uint32_t rows, cols;
    /* U [rows][rank] f16, then V [cols][rank] f16                          */
} waste_lowrank_hdr;

static_assert(sizeof(waste_lowrank_hdr) == 20, "lowrank header layout");

/* ---- usage.waste (append-only) ----------------------------------------- */
/* Runtime routing stats for the LFRU pinner + pilot/COUPLE prefetch.
 * Entry stream after a 16-byte header; LFRU counters + decayed recency.   */

typedef struct {
    uint32_t magic;          /* WASTE_MAGIC_USAGE                           */
    uint32_t version;
    uint64_t total_tokens;
} waste_usage_hdr;

typedef struct {
    uint16_t layer;
    uint16_t expert_id;
    uint32_t hits;           /* decayed hit count (LFRU frequency term)     */
    uint32_t last_seen;      /* token clock (recency tiebreak)              */
    uint32_t next_layer_top; /* most-coupled expert next layer (COUPLE)     */
} waste_usage_ent;

static_assert(sizeof(waste_usage_ent) == 16, "usage entry layout");

#endif /* WASTE_FORMAT_H */
