from contextlib import contextmanager
import gc

import torch


def create_tensors():
    return {
        "test1": torch.randn((151936, 2560), dtype=torch.bfloat16, requires_grad=False),
        "test2": torch.randn((2560, 4096), dtype=torch.bfloat16, requires_grad=False),
        "test3": torch.randn((2560, 1024), dtype=torch.bfloat16, requires_grad=False),
        "test4": torch.randn((2560, 9728), dtype=torch.bfloat16, requires_grad=False),
        "test5": torch.randn((1, 2560), dtype=torch.bfloat16, requires_grad=False),
    }


def get_cuda_tensors():
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensors.append(obj)
        except Exception:
            pass
    return tensors


@contextmanager
def search_tensors():
    before = {id(t): t for t in get_cuda_tensors()}
    before_mem = torch.cuda.memory_allocated()

    yield

    after_mem = torch.cuda.memory_allocated()
    after = {id(t): t for t in get_cuda_tensors()}
    new_tensors = {k: v for k, v in after.items() if k not in before}
    print(f"\n🔍 发现 {len(new_tensors)} 个新增 CUDA 张量：")
    total_bytes = 0
    for i, (tid, t) in enumerate(new_tensors.items()):
        numel = t.numel()
        element_size = t.element_size()  # 每个元素占多少字节（如 float32 → 4）
        size_bytes = numel * element_size
        total_bytes += size_bytes
        print(
            f"  [{i+1}] id={tid} | shape={list(t.shape)} | dtype={t.dtype} | "
            f"numel={numel:>8} | size={size_bytes:>8} B ({size_bytes/1024:.1f} KB)| total_memory={after_mem-before_mem} B"
        )


def test_my_method(test_tensors):
    """逐个分配到 GPU"""
    for key in test_tensors:
        print(f"---正在加载{key}---")
        test_tensors[key] = test_tensors[key].to("cuda")
    return test_tensors


def test_hf_method(test_tensors):
    """先预热再分配"""
    big = torch.empty([160000, 2700], dtype=torch.bfloat16, device="cuda")
    del big
    for key in test_tensors:
        print(f"---正在加载{key}---")
        test_tensors[key] = test_tensors[key].to("cuda")
    return test_tensors


def test_torch_allocated(test_tensors):
    """测试Pytorch如何分配内存"""
    for key in test_tensors:
        print(f"---正在加载{key}---")
        before_reserved = torch.cuda.memory_reserved()
        before_allocated = torch.cuda.memory_allocated()
        test_tensors[key] = test_tensors[key].to("cuda")
        after_reserved = torch.cuda.memory_reserved()
        after_allocated = torch.cuda.memory_allocated()
        print(
            f"张量理论内存 : {test_tensors[key].nbytes}, 实际占用内存 : {after_allocated-before_allocated}, 实际分配内存 : {after_reserved-before_reserved}"
        )
    return test_tensors


def test_warmup(test_tensors):
    """
    通过 GPU 虚拟地址测试张量是否落在预热大内存块内。
    """
    warmup_shape = (165000, 2700)
    big = torch.empty(warmup_shape, dtype=torch.bfloat16, device="cuda")

    warmup_start = big.data_ptr()
    warmup_size = big.nbytes
    warmup_end = warmup_start + warmup_size

    print(f"  预热块地址范围: [{warmup_start:#x}, {warmup_end:#x})")
    print(f"  预热块大小: {warmup_size / 1024**2:.1f} MB")

    del big

    print("\n分配测试张量并检查地址")
    all_in_warmup = True
    for key in test_tensors:

        test_tensors[key] = test_tensors[key].to("cuda")
        t = test_tensors[key]

        addr = t.data_ptr()
        size = t.nbytes
        end = addr + size

        in_warmup = (warmup_start <= addr) and (end <= warmup_end)
        status = "✅ 在预热块内" if in_warmup else "❌ 在预热块外"

        print(f"  {key}:")
        print(f"    地址范围: [{addr:#x}, {end:#x}] | {status}")

        if not in_warmup:
            all_in_warmup = False

    if all_in_warmup:
        print("所有测试张量的 GPU 地址均落在预热大内存块内！")
    else:
        print("部分张量分配在预热块之外，可能存在碎片或对齐问题。")


def main():

    print("=" * 20 + "测试自己函数" + "=" * 20)

    test_tensors1 = create_tensors()
    with search_tensors():
        result_tesnors1 = test_my_method(test_tensors1)
    del test_tensors1, result_tesnors1
    torch.cuda.empty_cache()

    print("=" * 20 + "测试显存预热" + "=" * 20)

    test_tensors2 = create_tensors()
    with search_tensors():
        result_tesnors2 = test_hf_method(test_tensors2)
    del test_tensors2, result_tesnors2
    torch.cuda.empty_cache()

    print("=" * 20 + "测试Pytorch分配策略" + "=" * 20)

    test_tensors3 = create_tensors()
    result_tesnors3 = test_torch_allocated(test_tensors3)
    del test_tensors3, result_tesnors3
    torch.cuda.empty_cache()

    print("=" * 20 + "测试预热功能" + "=" * 20)

    test_tensors4 = create_tensors()
    result_tesnors4 = test_warmup(test_tensors4)
    del test_tensors4, result_tesnors4
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
