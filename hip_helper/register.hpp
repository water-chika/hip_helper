#pragma once

#include <hip/hip_runtime.h>

#include "cpp_helper.hpp"

namespace hip_helper{

template<typename T=uint64_t>
    requires cpp_helper::same_as_one_of<T, uint32_t, uint64_t>
struct get_exec{
    __device__
    static inline auto invoke() {
        uint64_t exec;
        asm("s_mov_b64 %0, exec" : "=s"(exec));
        return exec;
    }

    __device__
    get_exec() {
        m_res = invoke();
    }

    __device__
    operator T() {
        return m_res;
    }

    T m_res;
};

__device__
inline auto get_pc() {
    uint64_t pc;
    asm("s_getpc_b64 %0" : "=s"(pc));
    return pc;
}

template<uint32_t HW_REG_INDEX>
__device__
inline uint32_t get_reg_value() {
    uint32_t v;
    asm("s_getreg_b32 %0, hwreg(%1, 0, 32)" : "=s"(v) : "i"(HW_REG_INDEX));
    return v;
}

template<uint32_t V, uint32_t... Vs>
struct parameters_call {
    __device__
    static inline void call(uint32_t* out) {
        out[V] = get_reg_value<V>();
        parameters_call<Vs...>::call(out);
    }
};
template<uint32_t V>
struct parameters_call<V> {
    __device__
    static inline void call(uint32_t* out) {
        out[V] = get_reg_value<V>();
    }
};

template<uint32_t I, uint32_t... Indices>
__global__
void device_get_reg_value(uint32_t* out) {
    parameters_call<I, Indices...>::call(out);
}

template<uint32_t N, uint32_t... Ns>
auto get_reg_values() {
    constexpr auto reg_indices = std::array{N, Ns...};
    auto reg_values = std::array<uint32_t, reg_indices.size()>{};
    {
        auto v = hip_helper::device::array<uint32_t, reg_values.size()>{};
        device_get_reg_value<N, Ns...><<<dim3(1,1,1),dim3(1,1,1),0>>>(v.data());
        reg_values = v;
    }
    return std::pair{reg_indices, reg_values};
}

}
