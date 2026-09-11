#ifndef _SS_FILEIO_H_
#define _SS_FILEIO_H_
#include "LargeFile.h"
#include <iostream>
#include <vector>
#include "CString.h"
# include "IncAng.h"

class BigEndianStream;
class SsSig;

class SsInAcAng : public IncAng {
    public:
        SsIncAng();
        ~SsIncAng();

        void Fetch(SsSig &);
        void Apply(SsSig &);

    protected:
        virtual void ReadFromAdvFeature(const AdvancedFeature & thisAdv);
        virtual void WriteToAdvFeature(AdvancedFeature & thisAdv) const;

};