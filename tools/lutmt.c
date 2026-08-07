/* SPDX-License-Identifier: Apache-2.0
 * Why does the VQ4P kernel not scale inside the engine?
 *
 * Same two kernels as tools/lutbw.c, but driven through the engine's own
 * waste_parallel_for, so the dispatch being tested is the real one. Thread
 * count and min_chunk come from argv, because the pool is a process-wide
 * singleton and cannot be resized once initialised.
 *
 *   cc -O3 -mcpu=native -I../../../../../Users/marco/GitHub/waste/src \
 *      -o lutmt lutmt.c -lm -lpthread
 *   ./lutmt <threads> <min_chunk_tiles> [copies]
 *
 * `copies` rotates over several index buffers so each pass reads bytes the
 * last one did not, the way the engine sees a fresh expert record every
 * time rather than the same hot buffer twenty times over.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <arm_neon.h>

#include "threads.h"

/* waste_parallel_for with the chunk chosen by the caller instead of
 * ceil(n/nthreads). Same pool, same stealing loop — the only variable is
 * how many tasks the work is cut into. */
static void my_parallel_for(int n, int chunk, waste_range_fn fn, void *arg)
{
    if (g_pool.nthreads <= 1 || n <= chunk) { fn(0, n, arg); return; }
    pthread_mutex_lock(&g_pool.mu);
    g_pool.fn=fn; g_pool.arg=arg; g_pool.n=n; g_pool.chunk=chunk;
    g_pool.next_chunk=0; g_pool.active=g_pool.nthreads-1; g_pool.epoch++;
    pthread_cond_broadcast(&g_pool.start);
    pthread_mutex_unlock(&g_pool.mu);
    for (;;) {
        pthread_mutex_lock(&g_pool.mu);
        const int b=g_pool.next_chunk;
        if (b>=n) { pthread_mutex_unlock(&g_pool.mu); break; }
        g_pool.next_chunk=b+chunk;
        pthread_mutex_unlock(&g_pool.mu);
        int e=b+chunk; if (e>n) e=n;
        fn(b,e,arg);
    }
    pthread_mutex_lock(&g_pool.mu);
    while (g_pool.active>0) pthread_cond_wait(&g_pool.done,&g_pool.mu);
    pthread_mutex_unlock(&g_pool.mu);
}

#define M_ROWS 3072            /* K3 gate: rows            */
#define NV     448             /* K3 gate: 3584/8          */
#define TILE   64

static double now(void)
{ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

typedef struct { float *y; const uint8_t *idx; const float *lut;
                 const int8_t *lut8; const float *sc; float ls; } arg_t;

static void rowsA(int b, int e, void *p)
{
    const arg_t *a = (const arg_t *)p;
    const int st=3, en=256;
    float acc[TILE];
    for (int r0=b; r0<e; r0+=TILE) {
        memset(acc,0,sizeof acc);
        for (int v=0; v<NV; v++) {
            const float *blk=a->lut+(size_t)v*st*en, *b1=blk+en, *b2=blk+2*en;
            const uint8_t *ix=a->idx+((size_t)(r0/TILE)*NV+v)*TILE*st;
            for (int r=0; r+8<=TILE; r+=8, ix+=24) {
                uint32_t w0,w1,w2,w3,w4,w5;
                memcpy(&w0,ix,4);    memcpy(&w1,ix+4,4);
                memcpy(&w2,ix+8,4);  memcpy(&w3,ix+12,4);
                memcpy(&w4,ix+16,4); memcpy(&w5,ix+20,4);
                acc[r]  +=blk[w0&0xff]       +b1[(w0>>8)&0xff] +b2[(w0>>16)&0xff];
                acc[r+1]+=blk[w0>>24]        +b1[w1&0xff]      +b2[(w1>>8)&0xff];
                acc[r+2]+=blk[(w1>>16)&0xff] +b1[w1>>24]       +b2[w2&0xff];
                acc[r+3]+=blk[(w2>>8)&0xff]  +b1[(w2>>16)&0xff]+b2[w2>>24];
                acc[r+4]+=blk[w3&0xff]       +b1[(w3>>8)&0xff] +b2[(w3>>16)&0xff];
                acc[r+5]+=blk[w3>>24]        +b1[w4&0xff]      +b2[(w4>>8)&0xff];
                acc[r+6]+=blk[(w4>>16)&0xff] +b1[w4>>24]       +b2[w5&0xff];
                acc[r+7]+=blk[(w5>>8)&0xff]  +b1[(w5>>16)&0xff]+b2[w5>>24];
            }
        }
        for (int r=0;r<TILE;r++) a->y[r0+r]=acc[r]*a->sc[r0+r];
    }
}

static void rowsD(int b, int e, void *p)
{
    const arg_t *a = (const arg_t *)p;
    int32_t acc[TILE];
    const uint8x16_t m3f=vdupq_n_u8(0x3f), m0f=vdupq_n_u8(0x0f), m03=vdupq_n_u8(0x03);
    for (int r0=b; r0<e; r0+=TILE) {
        int32x4_t A[16];
        for (int i=0;i<16;i++) A[i]=vdupq_n_s32(0);
        for (int v0=0; v0<NV; v0+=32) {
            const int ve = v0+32>NV?NV:v0+32;
            int16x8_t s[8];
            for (int i=0;i<8;i++) s[i]=vdupq_n_s16(0);
            for (int v=v0; v<ve; v++) {
                const int8_t *blk=a->lut8+(size_t)v*4*64;
                int8x16x4_t T0,T1,T2,T3;
                for (int k=0;k<4;k++){
                    T0.val[k]=vld1q_s8(blk+   0+k*16); T1.val[k]=vld1q_s8(blk+ 64+k*16);
                    T2.val[k]=vld1q_s8(blk+128+k*16);  T3.val[k]=vld1q_s8(blk+192+k*16);
                }
                const uint8_t *ix=a->idx+((size_t)(r0/TILE)*NV+v)*TILE*3;
                for (int g=0; g<4; g++) {
                    const uint8x16x3_t I=vld3q_u8(ix+g*48);
                    const uint8x16_t j0=vandq_u8(I.val[0],m3f);
                    const uint8x16_t j1=vorrq_u8(vshrq_n_u8(I.val[0],6),vshlq_n_u8(vandq_u8(I.val[1],m0f),2));
                    const uint8x16_t j2=vorrq_u8(vshrq_n_u8(I.val[1],4),vshlq_n_u8(vandq_u8(I.val[2],m03),4));
                    const uint8x16_t j3=vshrq_n_u8(I.val[2],2);
                    int8x16_t r =vaddq_s8(vqtbl4q_s8(T0,j0),vqtbl4q_s8(T1,j1));
                    int8x16_t r2=vaddq_s8(vqtbl4q_s8(T2,j2),vqtbl4q_s8(T3,j3));
                    s[g*2]  =vaddw_s8(s[g*2],  vget_low_s8(r));
                    s[g*2+1]=vaddw_s8(s[g*2+1],vget_high_s8(r));
                    s[g*2]  =vaddw_s8(s[g*2],  vget_low_s8(r2));
                    s[g*2+1]=vaddw_s8(s[g*2+1],vget_high_s8(r2));
                }
            }
            for (int i=0;i<8;i++){
                A[i*2]  =vaddw_s16(A[i*2],  vget_low_s16(s[i]));
                A[i*2+1]=vaddw_s16(A[i*2+1],vget_high_s16(s[i]));
            }
        }
        for (int i=0;i<16;i++) vst1q_s32(acc+i*4,A[i]);
        for (int r=0;r<TILE;r++) a->y[r0+r]=(float)acc[r]*a->ls*a->sc[r0+r];
    }
}

int main(int argc, char **argv)
{
    const int nthr   = argc>1 ? atoi(argv[1]) : 1;
    const int mchunk = argc>2 ? atoi(argv[2]) : 1;   /* in TILEs */
    const int copies = argc>3 ? atoi(argv[3]) : 24;  /* index buffers */
    const size_t nidx=(size_t)(M_ROWS/TILE)*NV*TILE*3;

    uint8_t **idx = malloc(sizeof(void*)*copies);
    for (int c=0;c<copies;c++) idx[c]=aligned_alloc(64,nidx);
    float *lutf=aligned_alloc(64,(size_t)NV*3*256*sizeof(float));
    int8_t *lut8=aligned_alloc(64,(size_t)NV*4*64);
    float *sc=aligned_alloc(64,M_ROWS*sizeof(float));
    float *y=aligned_alloc(64,M_ROWS*sizeof(float));
    srandom(1234);
    for (int c=0;c<copies;c++)
        for (size_t i=0;i<nidx;i++) idx[c][i]=(uint8_t)(random()&0xff);
    for (size_t i=0;i<(size_t)NV*3*256;i++) lutf[i]=(float)((random()%2001)-1000)/1000.0f;
    for (size_t i=0;i<(size_t)NV*4*64;i++) lut8[i]=(int8_t)((random()%127)-63);
    for (int i=0;i<M_ROWS;i++) sc[i]=0.01f;

    {   /* honours WASTE_CPUS, so a placement measured here is the
           placement the engine would use */
        waste_cpumask cpus;
        const int cr = waste_cpus_resolve(NULL, &cpus);
        waste_pool_init(nthr, cr == WASTE_CPUS_OK ? &cpus : NULL);
    }
    static volatile double sink;
    arg_t a = { y, NULL, lutf, lut8, sc, 1.0f/127.0f };
    const int REP = 40;

    /* A */
    a.idx = idx[0]; my_parallel_for(M_ROWS, TILE*mchunk, rowsA, &a);
    double t=now();
    for (int i=0;i<REP;i++){ a.idx=idx[i%copies];
        my_parallel_for(M_ROWS, TILE*mchunk, rowsA, &a); }
    const double ta=(now()-t)/REP;
    { double s2=0; for(int i=0;i<M_ROWS;i++) s2+=y[i]; sink=s2; }

    /* D */
    a.idx = idx[0]; my_parallel_for(M_ROWS, TILE*mchunk, rowsD, &a);
    t=now();
    for (int i=0;i<REP;i++){ a.idx=idx[i%copies];
        my_parallel_for(M_ROWS, TILE*mchunk, rowsD, &a); }
    const double td=(now()-t)/REP;
    { double s2=0; for(int i=0;i<M_ROWS;i++) s2+=y[i]; sink=s2; }

    const double mb=(double)nidx/1048576.0;
    const int tasks = (M_ROWS + TILE*mchunk - 1)/(TILE*mchunk);
    printf("thr %2d  chunk %2d tiles (%2d task)  A %7.3f ms %6.1f GB/s   "
           "D %7.3f ms %6.1f GB/s   D/A %5.2fx\n",
           nthr, mchunk, tasks,
           ta*1e3, mb/1024.0/ta, td*1e3, mb/1024.0/td, ta/td);
    return 0;
}
