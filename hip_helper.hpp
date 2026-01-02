#pragma once
#include <hip/hip_runtime.h>

#include "hip_helper/memory.hpp"

namespace hip_helper{

template<typename T, typename... Params>
struct hybrid_call {
    using Result = std::invoke_result_t<decltype(T::invoke), Params...>;

    __global__
    static void entry(auto* result, Params... params) {
        *result = T{params...};
    }
    static inline auto call(Params... params) {
        auto device_res = hip_helper::device::array<Result, 1>{};
        entry<<<dim3(1,1,1),dim3(1,1,1),0>>>(device_res.data(), params...);
        return device_res[0];
    }

    inline hybrid_call(Params... params) {
        m_res = call(params...);
    }

    __device__
    inline hybrid_call(Params... params) {
        m_res = T{params...};
    }

    inline operator Result() {
        return m_res;
    }

    Result m_res;
};

}
