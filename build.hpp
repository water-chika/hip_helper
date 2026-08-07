#pragma once

namespace hip_helper_build {

void add(build::builder& builder, std::filesystem::path path) {
    builder.add_library("hip_helper", absolute(path).string(), "/opt/rocm/include");
}

}
