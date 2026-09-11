#ifndef _CSTRING_H_ 
#define _CSTRING_H_

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>

class CString
{
    public:

        CString();
        CString(const char);
        CString (const char *);
        CString(const CString &);
        CString(CString &&) noexcept;
        CString(const CString *);
        CString(const std::string);

        inline virtual ~CString();

        CString & operator =(const char);
        CString & operator =(const char *);
        CString & operator =(const CString &); 
        CString & operator =(const std::string &);
        CString & operator =(CString &&) noexcept;
        
        CString & operator +=(const char);
        CString & operator +=(const char *);
        CString & operator +=(const CString &);
        inline CString & operator +=(const bool);
        inline CString & operator +=(const double);
        inline CString & operator +=(const float);
        inline CString & operator +=(const int);
        inline CString & operator +=(const unsigned int);
        CString & Add(const double, const int prec=15);

        inline operator char *();
        inline operator char *() const;
        inline operator const char *();
        inline operator const char *() const;
        inline static std::string ToStdString(CString const & cstring);

        inline bool operator ==(const char) const;
        inline bool operator ==(const char *) const;
        inline bool operator ==(const CString &) const;
    #if defined(_SGI_SOURCE) && defined(_SYSTYPE_SVR4)
        inline bool operator == (char *) const;

    #endif
}