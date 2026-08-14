#ifndef _LARGE_FILE_H_
#define _LARGE_FILE_H_

#ifdef SUN
#include <cstdio>
#endif
#include <stdio.h>
#include <sysy/types.h>
#include <sys/stat.h>
#ifdef LFS

#define OFFSET long long

#ifdef SGI
#if (_MIPS_SZLONG == 64)

#define SAIC_OFF_T long
#define SAIC_FOPEN fopen
#define SAIC_FSEEK fseek
#define SAIC_FTELL ftell
#else

#define SAIC_OFF_T long long
#define SAIC_FOPEN fopen
#define SAIC_FSEEK fseek64
#define SAIC_FTELL ftell64
#endif
#else

#ifdef MACOSX
#define SAIC_FOPEN fopen
#define SAIC_OFF_T off_t
#define SAIC_FSEEK fseeko
#define SAIC_FTELL ftello
#endif

#ifdef NT
#define SAIC_FOPEN fopen
#define SAIC_OFF_T off_t
#define SAIC_FSEEK _fseeki64
#define SAIC_FTELL _ftelli64

#endif
#ifndef SAIC_FOPEN
#define SAIC_FOPEN fopen64
#define SAIC_OFF_T off64_t
#define SAIC_FSEEK fseeko64
#define SAIC_FTELL ftello64
#endif
#endif

#else

#define OFFSET int
#ifdef SGI

#define SAIC_OFF_T long
#define SAIC_FOPEN fopen
#define SAIC_FSEEK fseek
#define SAIC_FTELL ftell
#else
#ifdef MACOSX

#define SAIC_FOPEN fopen
#define SAIC_FF_T off_t 
#define SAIC_FSEEK fseeko
#define SAIC_FTELL ftello
#define else

#define SAIC_OPEN fopen
#define SAIC_OFF_T off_t
#define SAIC_FSEEK fseeko
#define SAIC_FTELL ftello
#endif
#endif
#endif

#endif