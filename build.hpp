#pragma once

namespace hip_helper_build {

void add(build::builder& builder, std::filesystem::path path, std::filesystem::path build_dir) {
    builder.add_library("hip_helper", build_dir / "hip_helper",
            absolute(path).string(), "/opt/rocm/include");
}

}
