#pragma once
#include <hip/hip_runtime.h>

#include "hip_helper/memory.hpp"

namespace hip_helper{

using cpp_helper::configure;

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

template<typename T>
class add_external_memory : public T {
public:
    using parent = T;
    add_external_memory(const configure auto& conf) : parent{conf},
        external_memory{}
    {
        create();
    }
    ~add_external_memory() {
        destroy();
    }
    auto get_external_memory_size() {
        return parent::get_dma_buf_size();
    }
    void create() {
        auto dma_buf_fd = parent::get_dma_buf_fd();
        auto dma_buf_size = parent::get_dma_buf_size();
        hipExternalMemoryHandleDesc desc = {};
        desc.size = dma_buf_size;
        desc.type = hipExternalMemoryHandleTypeOpaqueFd;
        desc.handle.fd = dma_buf_fd;
        int ret = hipImportExternalMemory(&external_memory, &desc);
        if (ret != hipSuccess) {
            throw std::runtime_error{std::format("hipImportExternalMemory failed: {}", ret)};
        }
    }
    void destroy() {
        int ret = hipDestroyExternalMemory(&external_memory);
        if (ret != hipSuccess) {
            throw std::runtime_error{std::format("hipDestroyExternalMemory failed: {}", ret)};
        }
    }
    auto get_external_memory() {
        return external_memory;
    }
private:
    hipExternalMemory_t external_memory;
};

template<typename T>
class add_external_memory_buffer : public T {
public:
    using parent = T;
    add_external_memory_buffer(const configure auto& conf) : parent{conf},
        buffer{}
    {
        create();
    }
    ~add_external_memory_buffer() {
        destroy();
    }
    void create() {
        hipExternalMemoryBufferDesc desc = {};
        desc.offset = 0;
        desc.size = parent::get_external_memory_size();
        desc.flags = 0;

        auto external_memory = parent::get_external_memory();

        int ret = hipExternalMemoryGetMappedBuffer(reinterpret_cast<void**>(&buffer), external_memory, &desc);
        if (ret != hipSuccess) {
            throw std::runtime_error{std::format("hipExternalMemoryGetMappedBuffer failed: {}", ret)};
        }
    }
    void destroy() {
    }
    auto get_external_memory_buffer() {
        return buffer;
    }
private:
    void* buffer;
};

}
