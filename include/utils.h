#ifndef UTILS_H_
#define UTILS_H_

#include <stdio.h>
#include <cinttypes>
#include <string>
#include "GAP/pvector.h"

#define  EPSILON  0.001


template <typename T>
struct ErrorTolerantEqual:
public std::binary_function< T, T, bool >
{
    ErrorTolerantEqual(const T & myepsilon):epsilon(myepsilon) {};
    inline bool operator() (const T & a, const T & b) const
    {
        // According to the IEEE 754 standard, negative zero and positive zero should
        // compare as equal with the usual (numerical) comparison operators, like the == operators of C++
        
        if(a == b)      // covers the "division by zero" case as well: max(a,b) can't be zero if it fails
            return true;    // covered the integral numbers case
        
        return ( std::abs(a - b) < epsilon || (std::abs(a - b) / std::max(std::abs(a), std::abs(b))) < epsilon ) ;
    }
    T epsilon;
};


// TODO: just a temporary solution
// use a better prefix sum (see mtspgemm code)
template<typename T, typename SUMT>
static void ParallelPrefixSum(const pvector<T> &degrees, pvector<SUMT>& prefix)
{
    const size_t block_size = 1<<20;
    const size_t num_blocks = (degrees.size() + block_size - 1) / block_size;
    pvector<SUMT> local_sums(num_blocks);
#pragma omp parallel for
    for (size_t block=0; block < num_blocks; block++) {
        SUMT lsum = 0;
        size_t block_end = std::min((block + 1) * block_size, degrees.size());
        for (size_t i=block * block_size; i < block_end; i++)
            lsum += degrees[i];
        local_sums[block] = lsum;
    }
    pvector<SUMT> bulk_prefix(num_blocks+1);
    SUMT total = 0;
    for (size_t block=0; block < num_blocks; block++) {
        bulk_prefix[block] = total;
        total += local_sums[block];
    }
    bulk_prefix[num_blocks] = total;
    if(prefix.size() < (degrees.size() + 1))
        prefix.resize(degrees.size() + 1);
#pragma omp parallel for
    for (size_t block=0; block < num_blocks; block++) {
        SUMT local_total = bulk_prefix[block];
        SUMT block_end = std::min((block + 1) * block_size, degrees.size());
        for (size_t i=block * block_size; i < block_end; i++) {
            prefix[i] = local_total;
            local_total += degrees[i];
        }
    }
    prefix[degrees.size()] = bulk_prefix[num_blocks];
}



#endif  // UTILS_H_
