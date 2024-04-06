def get_cuda_dtype(name: str):
    return {
        "bfloat16": "__nv_bfloat16",
        "float16": "__half",
        "float32": "float",
        "float64": "double",
        "int8": "std::int8_t",
        "int16": "std::int16_t",
        "int32": "std::int32_t",
        "int64": "std::int64_t",
    }[name]
