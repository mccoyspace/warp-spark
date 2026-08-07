/* SPDX-License-Identifier: Apache-2.0
 * Does the VQ4P kernel's advantage survive a working set that does not fit
 * in cache? The original bench ran 20 passes over a 3.94 MB index buffer,
 * hot in L2. The engine streams ~17 GB of index per token from DRAM.
 *
 * Same two kernels, one pass, M settable so the index stream can be sized
 * from "fits in L2" to "definitely does not".
 *
 *   cc -O3 -mcpu=native -DM_ROWS=196608 -o lutbw lutbw.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <arm_neon.h>

#ifndef M_ROWS
#define M_ROWS 3072
#endif
#define NV   448
#define TILE 64

static double now(void)
{ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }

/* A: VQ3R today — 3 stages x 256 entries, fp32 table, scalar gather */
static void kern_A(float *y, const uint8_t *idx, const float *lut, const float *sc)
{
    const int st=3, en=256;
    float acc[TILE];
    for (int r0=0; r0<M_ROWS; r0+=TILE) {
        memset(acc,0,sizeof acc);
        for (int v=0; v<NV; v++) {
            const float *blk = lut + (size_t)v*st*en, *b1=blk+en, *b2=blk+2*en;
            const uint8_t *ix = idx + ((size_t)(r0/TILE)*NV + v)*TILE*st;
            for (int r=0; r+8<=TILE; r+=8, ix+=24) {
                uint32_t w0,w1,w2,w3,w4,w5;
                memcpy(&w0,ix,4);      memcpy(&w1,ix+4,4);
                memcpy(&w2,ix+8,4);    memcpy(&w3,ix+12,4);
                memcpy(&w4,ix+16,4);   memcpy(&w5,ix+20,4);
                acc[r]  +=blk[w0&0xff]        +b1[(w0>>8)&0xff] +b2[(w0>>16)&0xff];
                acc[r+1]+=blk[w0>>24]         +b1[w1&0xff]      +b2[(w1>>8)&0xff];
                acc[r+2]+=blk[(w1>>16)&0xff]  +b1[w1>>24]       +b2[w2&0xff];
                acc[r+3]+=blk[(w2>>8)&0xff]   +b1[(w2>>16)&0xff]+b2[w2>>24];
                acc[r+4]+=blk[w3&0xff]        +b1[(w3>>8)&0xff] +b2[(w3>>16)&0xff];
                acc[r+5]+=blk[w3>>24]         +b1[w4&0xff]      +b2[(w4>>8)&0xff];
                acc[r+6]+=blk[(w4>>16)&0xff]  +b1[w4>>24]       +b2[w5&0xff];
                acc[r+7]+=blk[(w5>>8)&0xff]   +b1[(w5>>16)&0xff]+b2[w5>>24];
            }
        }
        for (int r=0;r<TILE;r++) y[r0+r]=acc[r]*sc[r0+r];
    }
}

/* D: VQ4P — 4 stages x 64 entries, int8 table, one vqtbl4q per stage */
static void kern_D(float *y, const uint8_t *idx, const int8_t *lut8,
                   const float *sc, const float *ls)
{
    int32_t acc[TILE];
    const uint8x16_t m3f=vdupq_n_u8(0x3f), m0f=vdupq_n_u8(0x0f), m03=vdupq_n_u8(0x03);
    for (int r0=0; r0<M_ROWS; r0+=TILE) {
        int32x4_t A[16];
        for (int i=0;i<16;i++) A[i]=vdupq_n_s32(0);
        for (int v0=0; v0<NV; v0+=32) {
            const int ve = v0+32>NV?NV:v0+32;
            int16x8_t s[8];
            for (int i=0;i<8;i++) s[i]=vdupq_n_s16(0);
            for (int v=v0; v<ve; v++) {
                const int8_t *blk = lut8 + (size_t)v*4*64;
                int8x16x4_t T0,T1,T2,T3;
                for (int k=0;k<4;k++){
                    T0.val[k]=vld1q_s8(blk+   0+k*16); T1.val[k]=vld1q_s8(blk+ 64+k*16);
                    T2.val[k]=vld1q_s8(blk+128+k*16);  T3.val[k]=vld1q_s8(blk+192+k*16);
                }
                const uint8_t *ix = idx + ((size_t)(r0/TILE)*NV + v)*TILE*3;
                for (int g=0; g<4; g++) {
                    const uint8x16x3_t I = vld3q_u8(ix+g*48);
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
        for (int r=0;r<TILE;r++) y[r0+r]=(float)acc[r]*ls[0]*sc[r0+r];
    }
}

int main(void)
{
    const size_t nidx=(size_t)(M_ROWS/TILE)*NV*TILE*3;
    uint8_t *idx=aligned_alloc(64,nidx);
    float *lutf=aligned_alloc(64,(size_t)NV*3*256*sizeof(float));
    int8_t *lut8=aligned_alloc(64,(size_t)NV*4*64);
    float *sc=aligned_alloc(64,M_ROWS*sizeof(float));
    float *y=aligned_alloc(64,M_ROWS*sizeof(float));
    float ls[1]={1.0f/127.0f};
    srandom(1234);
    for (size_t i=0;i<nidx;i++) idx[i]=(uint8_t)(random()&0xff);
    for (size_t i=0;i<(size_t)NV*3*256;i++) lutf[i]=(float)((random()%2001)-1000)/1000.0f;
    for (size_t i=0;i<(size_t)NV*4*64;i++) lut8[i]=(int8_t)((random()%127)-63);
    for (int i=0;i<M_ROWS;i++) sc[i]=0.01f;

    /* one pass each, cold-ish: the index stream is the thing being sized */
    static volatile double sink;
    double t=now(); kern_A(y,idx,lutf,sc);        const double ta=now()-t;
    { double s2=0; for (int i=0;i<M_ROWS;i++) s2+=y[i]; sink=s2; }
    t=now();        kern_D(y,idx,lut8,sc,ls);     const double td=now()-t;
    { double s2=0; for (int i=0;i<M_ROWS;i++) s2+=y[i]; sink=s2; }
    const double mb=(double)nidx/1048576.0;
    printf("indici %8.1f MB   A %8.3f ms   D %8.3f ms   speedup %5.2fx   "
           "(A %5.1f GB/s, D %5.1f GB/s)\n",
           mb, ta*1e3, td*1e3, ta/td,
           mb/1024.0/ta, mb/1024.0/td);
    return 0;
}
