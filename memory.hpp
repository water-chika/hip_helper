#pragma once

#include <stdexcept>

namespace hip_helper {

template<typename T>
auto hip_malloc(size_t size = 1) {
    void* ptr{};
    auto res = hipMalloc(&ptr, size);
    if (res != hipSuccess) {
        throw std::runtime_error{"hipMalloc failed"};
    }
    return reinterpret_cast<T*>(ptr);
}

void hip_free(auto* ptr) {
    hipFree(ptr);
}

enum class hip_memcpy_kind : uint32_t {
    host_to_host = hipMemcpyHostToHost,
    host_to_device = hipMemcpyHostToDevice,
    device_to_host = hipMemcpyDeviceToHost,
    device_to_device = hipMemcpyDeviceToDevice,
    based_on_address = hipMemcpyDefault,
    device_to_device_no_cu = hipMemcpyDeviceToDeviceNoCU,
};

void hip_memcpy(void* dst, const void* src, size_t size_in_bytes, hip_memcpy_kind kind) {
    auto res = hipMemcpy(dst, src, size_in_bytes, static_cast<hipMemcpyKind>(kind));
}

namespace device {

template<typename T, size_t N>
class array {
public:
    array() : m_ptr{hip_malloc<T>(N*sizeof(T))}
    {}
    array(const array& v) : m_ptr{hip_malloc<T>(N*sizeof(T))} {
        
    }
    array(array&& v) : m_ptr{v.m_ptr} {
        v.m_ptr = nullptr;
    }
    ~array() {
        if (m_ptr != nullptr) {
            //hip_free(m_ptr);
            m_ptr = nullptr;
        }
    }
    array& operator=(const std::array<T, N>& rhs) {
        hip_memcpy(m_ptr, rhs.data(), sizeof(T)*N, hip_memcpy_kind::host_to_device);
        return *this;
    }
    operator std::array<T, N>() {
        std::array<T,N> lhs;
        auto& rhs = *this;
        hip_memcpy(lhs.data(), rhs.data(), sizeof(T)*N, hip_memcpy_kind::device_to_host);
        return lhs;
    }
    auto size() {
        return N;
    }
    auto data() {
        return m_ptr;
    }
private:
    T* m_ptr;
};

}

}
